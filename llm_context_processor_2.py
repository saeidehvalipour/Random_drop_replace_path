import random
import logging
from pathlib import Path
import time
from datetime import datetime
from llm_explainer.llm_explainer import vllm_get_response
from hgcr_util.lazy_json_kv_loader import LazyJsonlAbstractLoader


class LLMContextProcessor:
    def __init__(self, sents_db, log_dir="./logs"):
        self.logger = self.setup_logger(log_dir)
        self.sents_db = sents_db

    @staticmethod
    def setup_logger(log_dir):
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir_path / f"llm_context_log_{timestamp}.txt"

        logger = logging.getLogger("LLM_Context_Logger")
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file, mode="w")
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def process_df_with_llm(self, df, max_iterations, k):
        # remember that we have template_prompt we are just upodating context appended into it in each iteration
        responses_per_row = []
        response_times_per_row = []
        prompts_per_row = []
        pmid_contexts_per_row = []
        pmid_changes_per_row = []

        for idx, row in df.iterrows():
            source_cui = row["subj_name"]
            target_cui = row["obj_name"]
            path_context_pmids = row["final_context_pmids"]
            selected_pmids = {edge: pmids[:k] for edge, pmids in path_context_pmids.items()}
            used_pmids = {edge: set(pmids[:k]) for edge, pmids in path_context_pmids.items()}
            iteration_responses = []
            iteration_response_times = []
            iteration_prompts = []
            iteration_pmid_contexts = []
            iteration_pmid_changes = []

            # Log new row processing, convienience to read .txt file
            self.logger.info("=" * 80)
            self.logger.info(f"Processing Row {idx}/{len(df)}: Source={source_cui}, Target={target_cui}")
            self.logger.info("=" * 80)

            for iteration in range(max_iterations):
                # Log the context before sending to LLM
                pmid_context = {edge: list(pmids)[:k] for edge, pmids in selected_pmids.items()}  # Ensure k PMIDs
                iteration_pmid_contexts.append(pmid_context)  # Store context for this iteration
                self.logger.info(f"Context for iteration {iteration + 1}: {pmid_context}")

                # Prepare the prompt
                prompt = self._prepare_prompt(source_cui, target_cui, selected_pmids)
                self.logger.info(f"Iteration {iteration + 1}/{max_iterations}: Full prompt sent to LLM:")
                self.logger.info(prompt)

                # Measure only LLM response time per Iteration, not per row; just sum htem up if need per row.
                try:
                    start_llm_time = time.time()
                    response = vllm_get_response(prompt)
                    response_time = time.time() - start_llm_time
                    iteration_response_times.append(response_time)
                    self.logger.info(f"Response received for Row {idx}, Iteration {iteration + 1}: {response}")
                    self.logger.info(f"Response time: {response_time:.2f} seconds")

                    # Log the full response using the logger
                    self.logger.info(f"Full response for Row {idx}, Iteration {iteration + 1}:\n{response}\n")
                except Exception as e:
                    response = f"Error: {str(e)}"
                    response_time = 0
                    iteration_response_times.append(response_time)
                    self.logger.error(f"Failed to get LLM response for Row {idx}, Iteration {iteration + 1}: {str(e)}")

                # Save response and prompt for this iteration
                iteration_response = {"iteration": iteration + 1, "response": response}
                iteration_responses.append(iteration_response)
                iteration_prompts.append(prompt)

                # Update context after receiving response, except for the last iteration
                if iteration < max_iterations - 1:
                    edge_to_modify = random.choice(list(selected_pmids.keys()))
                    pmid_change = self._replace_pmid(edge_to_modify, selected_pmids, path_context_pmids, used_pmids)
                    if pmid_change["added"] is None:
                        self.logger.info(f"Not Enough PMIDs to be found for Row {idx}. Stopping early.")
                        # no need to log as Error but need to drop that path if there is not enough pmid (most probable scenario for Sasha's case_study due to not covering space pmid's)
                        break
                    iteration_pmid_changes.append(pmid_change)
                    self._log_pmid_change(pmid_change)
                else:
                    # Log reaching max iteration
                    iteration_pmid_changes.append({"edge": "max_iteration", "dropped": "max_iteration", "added": "max_iteration"})
                    self.logger.info(f"Reached max iteration for Row {idx}.")

            # Store results for the row
            responses_per_row.append(iteration_responses)
            response_times_per_row.append(iteration_response_times)
            prompts_per_row.append(iteration_prompts)
            pmid_contexts_per_row.append(iteration_pmid_contexts)
            pmid_changes_per_row.append(iteration_pmid_changes)

        # Add new columns to the DataFrame
        df["respond_llm"] = responses_per_row
        df["respond_time_llm"] = response_times_per_row
        df["prompts_per_iteration"] = prompts_per_row
        df["pmid_contexts_per_iteration"] = pmid_contexts_per_row
        df["pmid_changes_per_iteration"] = pmid_changes_per_row
        return df

    def _replace_pmid(self, edge, selected_pmids, path_context_pmids, used_pmids):
        # Select a PMID to drop strategically
        pmid_to_drop = self._select_pmid_to_drop(selected_pmids[edge])
        remaining_pmids = [pmid for pmid in selected_pmids[edge] if pmid != pmid_to_drop]
        pmid_list = path_context_pmids[edge]  # Full list of PMIDs for this edge

        # Find the next PMID to replace due to they are already positioned in sorted manner
        for pmid in pmid_list:
            if pmid not in used_pmids[edge]:
                selected_pmids[edge] = remaining_pmids + [pmid]
                used_pmids[edge].add(pmid)
                return {"edge": edge, "dropped": pmid_to_drop, "added": pmid}
        # No replacement available, maintain current PMIDs
        selected_pmids[edge] = remaining_pmids + [pmid_to_drop]
        return {"edge": edge, "dropped": pmid_to_drop, "added": None}

    def _select_pmid_to_drop(self, pmids):
        # Placeholder for strategic selection logic
        # Currently selects randomly, but you can replace this with your logic
        return random.choice(pmids)

    def _log_pmid_change(self, pmid_change):
        if pmid_change["added"]:
            self.logger.info(
                f"Replaced PMID {pmid_change['dropped']} with {pmid_change['added']} in edge {pmid_change['edge']}"
            )
        else:
            self.logger.info(
                f"Dropped PMID {pmid_change['dropped']} from edge {pmid_change['edge']} with no replacement available"
            )

    def _prepare_prompt(self, source, target, selected_pmids):
        abstracts = []
        for edge, pmids in selected_pmids.items():
            for pmid in pmids:
                try:
                    abstract = self.sents_db[pmid]
                    if abstract:
                        abstracts.append(abstract)
                    else:
                        self.logger.error(f"No abstract found for PMID: {pmid}")
                        raise RuntimeError(f"No abstract found for PMID: {pmid}")
                except KeyError:
                    # sent_db sometimes play around by not finding that particular pmid
                    self.logger.error(f"PMID {pmid} not found in the database.")
                    raise RuntimeError(f"PMID {pmid} not found in the database.")

        if not abstracts:
            self.logger.error("No abstracts retrieved for the selected PMIDs.")
            raise RuntimeError("No abstracts retrieved for the selected PMIDs.")

        prompt_template = (
            "Based on the following scientific abstracts, please describe how an indirect relationship "
            "between {source} and {target} might exist. Consider the key findings, underlying mechanisms, "
            "and any intermediate entities or processes mentioned in the abstracts. Your explanation "
            "should connect these elements to form a coherent narrative that illustrates the possible "
            "indirect linkage between {source} and {target}."
        )
        return prompt_template.format(source=source, target=target) + "\n\n" + "\n\n".join(abstracts)
