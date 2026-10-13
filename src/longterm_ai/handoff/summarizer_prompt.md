# Summarizer prompt for a handoff record

Spec section 25. Send this to any model you choose, with the source reasoning and the task attached. Do not include the mandatory constraints in what you ask the model to write. Code copies those in afterward.

Prompt text:

You are writing a handoff record so another model can continue this task. Use only what is in the reasoning and the evidence list provided. Do not add anything new. Return JSON with these fields: goal, success_test, decisions, rejected_options, facts, open_questions, next_action, changed_state. Each decision has decision, reason, and evidence, where evidence is a list of pointers taken from the evidence list. Each fact has fact and evidence. If a field has nothing, return an empty list or an empty string. Do not guess. If something is unclear, put it in open_questions.
