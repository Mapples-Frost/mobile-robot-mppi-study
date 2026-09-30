"""The user-authorized English working-language contract shared by all three agents."""
POLICY_ID = 'RESEARCH_WORKING_LANGUAGE_EN_V1'
POLICY = '''RESEARCH_WORKING_LANGUAGE_EN_V1
The user now requires English as the sole working language between the research agents.
Write all newly authored scientific analyses, independent reviews, execution plans,
handoff messages, experiment descriptions, state narratives, research-log additions,
code comments and execution feedback in English. This applies to Opus, Astra and GPT-5.5.
This current instruction supersedes older requests for Chinese internal reports.
Historical Chinese reports and source excerpts remain evidence: read them normally,
retain exact quotations/identifiers when necessary, and explain their meaning in English.
Do not translate or rewrite old signed model blocks, raw results, checkpoint data,
frozen protocols, published reports or their hashes merely to change their language.
Use precise definitions, equations, units, numerical thresholds and primary-evidence
paths. English alone does not establish causal validity or eliminate ambiguity.
Keep verified findings, hypotheses, acceptance criteria and failure criteria distinct.
Models, reasoning effort, role authority, research scope, budgets and test isolation
remain unchanged. The separate final Chinese summary for the human user remains
authorized; it is not the internal analysis/handoff document.
'''


def system_text(text):
    # Only current authored instructions are normalized; historical assistant/tool blocks stay opaque.
    # Remove only our exact prior policy block; preserve later mission amendments.
    text = text.replace(POLICY, '').rstrip()
    for old, new in [
        ('substantive Chinese report', 'substantive English report'),
        ('Write final report in Chinese', 'Write final report in English'),
        ('complete Chinese audit report', 'complete English audit report'),
    ]:
        text = text.replace(old, new)
    return text.rstrip() + '\n\n' + POLICY


def responses_input(items):
    output = []
    system_seen = False
    for item in items:
        if isinstance(item, dict) and item.get('role') == 'system' and isinstance(item.get('content'), str):
            output.append(dict(item, content=system_text(item['content'])))
            system_seen = True
        else:
            output.append(item)
    if not system_seen:
        output.insert(0, dict(role='system', content=POLICY))
    return output
