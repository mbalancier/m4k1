"""
Populism Meets AI
https://arxiv.org/abs/2510.07458

Python conversion of the original R script.
Scores political speeches for populism (0.0–2.0) using the OpenRouter API.

Original authors: Eduardo Ryo Tamaki, Yujin J. Jung*, Levente Littvay
* Corresponding: yujinjuliajung@gmail.com

Please cite the paper above if you use this script.
"""

import requests
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================

OPENROUTER_API_KEY = "..YOUR KEY.."  # Get yours at https://openrouter.ai/keys

MODEL = "qwen/qwen3-235b-a22b-thinking-2507"
# See all models at https://openrouter.ai/models

PROVIDER_PREFERENCES = {
    "order": ["deepinfra/fp8"],
    "allow_fallbacks": False,
}

N_REPLICATIONS = 1
# Set to 5 for test-retest reliability

# ==============================================================================
# SPEECH INPUT — replace this with the speech you want to score
# ==============================================================================

speech = """
!!! REPLACE THIS WITH THE SPEECHES YOU WANT TO CODE !!!
"""

# Example speech (uncomment to test):
# speech = """
# Speaker Evans, President Hester ...
# """

# ==============================================================================
# PROMPTS (Chain-of-Thought training sequence)
# ==============================================================================

INTRO_SYSTEM = (
    "You are a researcher who quantifies populist and pluralist discourse in speeches "
    "by political figures. You speak every and any language necessary. The training we "
    "present here teaches you how to classify and quantify populist and pluralist "
    "discourse in speeches by political figures. This training uses a technique called "
    "'holistic grading'. It will teach you 1) the definition and necessary components "
    "of populism, 2) the process of holistic grading, and 3) give several real world "
    "examples of how we want you to read, think, and grade as you analyze speeches for "
    "populist discourse."
)

# Each element is {"user": "...", "assistant": "..."}
FEW_SHOT_EXAMPLES = [
    {
        "user": """What is populism? To explain this, we use the ideational approach to studying populism, which views populism as a Manichaean discourse that identifies good with a unified will of the people and evil with a conspiring elite.

Scholars who define populism ideationally use a variety of labels—referring to it as a political 'style,' a 'discourse,' a 'language,' an 'appeal', or a 'thin ideology'—but all of them see it as a set of ideas rather than as a set of actions isolated from their underlying meanings for leaders and participants. 
What are these ideas that constitute populist discourse? Analyses of populist discourse all highlight a series of common, rough elements of linguistic form and content that distinguish populism from other political discourses. 
First, populism is a Manichaean discourse because it assigns a moral dimension to everything, no matter how technical, and interprets it as part of a cosmic struggle between good and evil. History is not just proceeding toward some final conflict but has already arrived, and there can be no fence sitters in this struggle. 

Within this dualistic vision, the good has a particular identity: It is the will of the people. The populist notion of the popular will is essentially a crude version of Rousseau's general Will. The mass of individual citizens are the rightful sovereign; given enough time for reasoned discourse, they will come to a knowledge of their collective interest, and the government must be constructed in such a way that it can embody their will. The populist notion of the general Will ascribes particular virtue to the views and collective traditions of common, ordinary folk, who are seen as the overwhelming majority. The voice of the people is the voice of god— 'Vox populi, vox dei.' 

On the other side of this Manichaean struggle is a conspiring elite that has subverted the will of the people. As Hofstadter (1966) eloquently describes in a classic essay on the 'paranoid mentality' in American politics, for populists 'this enemy is clearly delineated: he is a perfect model of malice, a kind of amoral superman: sinister, ubiquitous, powerful, cruel, sensual, luxury-lovin'. Populism has a preoccupation with discovering and identifying this enemy, as this process is what helps negatively constitute the people.

In this sense, in a populist speech, the 'people' is created in a moral opposition to the 'elite.' This means that populism has three necessary elements that must ALL be present:
(1) a pure and virtuous people, 
(2) an evil conspiring elite bent on subverting the people's will, AND
(3) moral dualism that creates these groups in direct opposition (people = good, elite = evil).

Without all three elements working together, discourse is not truly populist.

What is the opposite of populism? While populism may have no single opposite, or even a true opposite at all, one discourse/frame that often comes up as incompatible or opposed to populism is pluralism. Indeed, in the training we will present to you, pluralism is treated as analogous to an opposite of populism such that one is rarely strongly populist and pluralist.

Political pluralism is a theory and practice of politics that recognizes and affirms the diversity of interests, values, and identities within a society, emphasizing that no single group should dominate the political process. It holds that political power is, and should be, distributed among multiple competing groups, each with legitimate input into decision-making. Political pluralism values liberal and democratic institutional mechanisms (like free elections, separation of powers, and legal protections for dissent) that ensure open participation and safeguard against authoritarianism. At its core, it assumes that conflict and disagreement are natural and productive elements of democratic life, best managed through negotiation, compromise, and institutional checks and balances.

This more accepting view strongly contrasts with populism in several ways. First, the view that there are many groups that can co-exists peacefully and legitimately clashes with the dualistic view of populism that perceives politics as an epic battle between two antagonistic groups, with the good people being the only legitimate force in politics. Furthermore, pluralist emphasis on good faith attempts to engage in honest debate similarly clash with populism's view that elite are evil and conspiring and should be destroyed, not treated to good faith debate. Finally, pluralism focus on democratic and institutional means to resolve conflict juxtaposes against the more radical and sometimes illiberal framing of populist leader, who often deride institutions that protect minority interests at the expense of the majority.""",

        "assistant": """There are three necessary components to populism discourse: (1) pure and virtuous people, (2) evil conspiring elite bent on subverting the people's will, and (3) moral dualism that positions people as 'good' in direct opposition to elite as 'evil'. As I score speeches, it's important to remember that all of these components need to be present. Populism increases in intensity when these are used consistently. Similarly, extensive use of pluralist rhetoric is often indicative of little to no populism, though the two frames can exist in moderate amounts.""",
    },
    {
        "user": """The technique we use to measure populism in political speeches is called holistic grading. Holistic grading, unlike standard techniques of content analysis (either human coded or computer based), asks readers to interpret whole texts rather than count content at the level of words or sentences.""",
        "assistant": """To effectively evaluate speeches for populist qualities, I'll use holistic grading — interpreting the overall meaning and tone of entire passages rather than counting specific words. True holistic grading requires moving beyond surface-level cues and engaging deeply with the structure, tone, and underlying worldview conveyed by the speech.""",
    },
    {
        "user": """We have developed a rubric to guide coders on how to grade speeches on a scale of 0 to 2, with higher values indicating higher levels of populism.

- 2.0: Extremely populist; all three necessary elements present consistently, few non-populist elements.
- 1.0: All three elements present but used inconsistently or tempered by non-populist elements.
- 0.0: Few or no populist elements; possibly strong pluralist framing.

Scores go to the tenths place; always round down.

A speech is only considered populist if it has:
(1) a pure and virtuous people,
(2) an evil conspiring elite bent on subverting the people's will, AND
(3) moral dualism that creates these groups in direct opposition.""",
        "assistant": " ",
    },
    {
        "user": """Read the speech and code the level of populism (0.0–2.0 to the tenths place):

<speech>
[Mugabe speech at WSSD, 2002 — see original for full text]
</speech>""",
        "assistant": "Grade: 1.7",
    },
    {
        "user": """Read the speech and code the level of populism (0-2 to the tenth place):

<speech>
[Tony Blair EBRD speech, 2004 — see original for full text]
</speech>""",
        "assistant": "Grade: 0.0",
    },
    {
        "user": """Read the speech and code the level of populism (0-2 to the tenth place):

<speech>
[Evo Morales speech, 2003 — see original for full text]
</speech>""",
        "assistant": "Grade: 2.0",
    },
    {
        "user": """Read the speech and code the level of populism (0-2 to the tenth place):

<speech>
[Stephen Harper Federal Accountability Act speech, 2005 — see original for full text]
</speech>""",
        "assistant": "Grade: 1.0",
    },
]

# ==============================================================================
# API CALL FUNCTION
# ==============================================================================

def make_openrouter_call(
    model: str,
    system_content: str,
    user_content: str,
    few_shot_examples: list = None,
    temperature: float = None,
    max_tokens: int = None,
    provider_preferences: dict = None,
    return_metadata: bool = False,
    app_name: str = None,
    app_url: str = None,
) -> str | dict:
    """
    Call the OpenRouter chat completions API.

    Parameters
    ----------
    model : str
        OpenRouter model string, e.g. "qwen/qwen3-235b-a22b-thinking-2507".
    system_content : str
        System prompt.
    user_content : str
        The final user message (the speech to score).
    few_shot_examples : list of dict, optional
        List of {"user": "...", "assistant": "..."} pairs inserted before
        the final user message.
    temperature : float, optional
        Sampling temperature (0–2).
    max_tokens : int, optional
        Maximum tokens in the response.
    provider_preferences : dict, optional
        OpenRouter provider routing preferences.
    return_metadata : bool
        If True, return a dict with content + metadata; otherwise return str.
    app_name / app_url : str, optional
        Attribution headers sent to OpenRouter.

    Returns
    -------
    str or dict
        The model's text response, or a metadata dict if return_metadata=True.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    }
    if app_name:
        headers["X-Title"] = app_name
    if app_url:
        headers["HTTP-Referer"] = app_url

    # Build message list
    messages = [{"role": "system", "content": system_content}]

    if few_shot_examples:
        for ex in few_shot_examples:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})

    messages.append({"role": "user", "content": user_content})

    # Build request body
    body: dict = {"model": model, "messages": messages}
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if provider_preferences:
        body["provider"] = provider_preferences

    response = requests.post(url, headers=headers, data=json.dumps(body))

    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        if return_metadata:
            return {
                "content": content,
                "model": result.get("model"),
                "usage": result.get("usage"),
                "finish_reason": result["choices"][0].get("finish_reason"),
                "response_id": result.get("id"),
            }
        return content
    else:
        error = response.json()
        msg = error.get("error", {}).get("message", "Unknown error")
        raise RuntimeError(
            f"OpenRouter API error {response.status_code}: {msg}"
        )


# ==============================================================================
# SCORING
# ==============================================================================

def score_speech(speech_text: str, n_replications: int = 1) -> list[str]:
    """
    Score a speech for populism, optionally running multiple replications.

    Parameters
    ----------
    speech_text : str
        The full text of the speech.
    n_replications : int
        Number of times to score the same speech (for reliability assessment).

    Returns
    -------
    list of str
        One response string per replication.
    """
    user_content = (
        "Read the speech and code the level of populism (0.0–2.0 to the tenths place):\n\n"
        f"<speech>\n{speech_text}\n</speech>"
    )

    results = []
    for i in range(n_replications):
        print(f"  Running replication {i + 1}/{n_replications}...")
        response = make_openrouter_call(
            model=MODEL,
            system_content=INTRO_SYSTEM,
            user_content=user_content,
            few_shot_examples=FEW_SHOT_EXAMPLES,
            provider_preferences=PROVIDER_PREFERENCES,
        )
        results.append(response)

    return results


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Populism Scorer — Populism Meets AI (arxiv: 2510.07458)")
    print("=" * 60)

    if "REPLACE THIS" in speech:
        print("\n⚠  No speech provided. Please set the `speech` variable.")
    else:
        print(f"\nScoring speech with model: {MODEL}")
        print(f"Replications: {N_REPLICATIONS}\n")

        responses = score_speech(speech, n_replications=N_REPLICATIONS)

        for idx, resp in enumerate(responses, 1):
            print(f"\n--- Replication {idx} ---")
            print(resp)