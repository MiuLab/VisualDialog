"""Prompt utils"""

def get_descriptor_prompt(diag_data):
    prompt_instr = "Please read the following dialogue context:"
    diag_start_tok = "<start of dialogue>"
    diag_end_tok = "</end of dialogue>"
    list_instr = [
        "- main subject: {simply list the answer by ','}",
        "- prominent objects in the foreground: {simply list the answer by ','}",
        "- background scene: {one background scene}",
        "- events: {simply list the answer by ','}",
        "- materials and attributes: {simply list the answer by ','}",
        "- actions: {simply list the answer by ','}",
        "- atmosphere or mood: {simply list the answer by ','}",
        "- lighting: {simply list the answer by ','}",
    ]
    answer_instr = "Based on the dialogue context, please describe the photograph shared by user 0. List the answer in JSON format (\"key\" (str): \"value\" (str))"

    diag_list = "\n".join(diag_data)
    ans_list = "\n".join(list_instr)
    
    prompt = prompt_instr + "\n\n" + diag_start_tok + "\n\n" + diag_list + "\n\n" + diag_end_tok + "\n\n" + answer_instr + "\n\n" + ans_list + "\n\n" + "Answers:\n\n{"

    return prompt


def get_guessing_prompt(diag_data):
    prompt_instr = "Please read the following dialogue context:"
    diag_start_tok = "<start of dialogue>"
    diag_end_tok = "</end of dialogue>"
    answer_instr = "Based on the dialogue context, please describe the photograph shared by user 0."
    ans_start_tok = "<start of answer>"

    diag_list = "\n".join(diag_data)
    
    prompt = prompt_instr + "\n\n" + diag_start_tok + "\n\n" + diag_list + "\n\n" + diag_end_tok + "\n\n" + answer_instr + "\n\n" + ans_start_tok + "\n\n"

    return prompt


def get_summarization_prompt(diag_data):
    prompt_instr = "Please read the following dialogue context:"
    diag_start_tok = "<start of dialogue>"
    diag_end_tok = "</end of dialogue>"
    diag_list = "\n".join(diag_data)
    answer_instr = "Based on the dialogue context, please summarize the information of user 0"

    diag_list = "\n".join(diag_data)
   
    prompt = prompt_instr + "\n\n" + diag_start_tok + "\n\n" + diag_list + "\n\n" + diag_end_tok + "\n\n" + answer_instr + "\n\n" +  "<start of answer>\n"

    return prompt

