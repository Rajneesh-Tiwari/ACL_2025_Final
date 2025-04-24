import json
import argparse
import random
import time
import re
# from tqdm import tqdm
# from unsloth import FastLanguageModel
# from vllm import SamplingParams

try:
    from unsloth import FastLanguageModel
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "unsloth==2025.3.9"])
    from unsloth import FastLanguageModel

try:
    from vllm import SamplingParams
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "vllm==0.7.3"])
    from vllm import SamplingParams

try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "tqdm"])
    from tqdm import tqdm

# Template definitions
QP_TEMPLATE = '''Given the following question, please list all original fragments in the question that represent conditions related to solving the problem. Identify specific constraints, rules, or requirements that are essential for solving the problem.

The question is:

{question}

Output the results in the following list format, do not output other information:

[
    "condition 1",
    "condition 2",
    ...
]
'''

CP_TEMPLATE = '''Given a question, conditions in the question, and a chain of thought that attempts to solve the problem, your task is to analyze the reasoning by:

1. Extracting 4-6 key statements from the chain of thought
2. For each statement, identifying the specific evidence from the chain of thought that supposedly supports it 
3. Rigorously verifying whether the evidence LOGICALLY AND SUFFICIENTLY supports the statement

When determining verification:
- Mark as "true" ONLY if the evidence DIRECTLY AND LOGICALLY supports the statement without requiring additional assumptions
- Mark as "false" if:
  * The evidence contradicts the statement
  * The evidence is incomplete or insufficient to reach the statement
  * The reasoning requires logical leaps not present in the evidence
  * The evidence is merely related to the statement but doesn't actually support it

Aim for a BALANCED assessment with both true and false verifications. Most chains of thought contain statements that are not fully supported by their evidence.

The question is:
{question}

The conditions are:
{conditions}

The chain of thought is:
{cot}

Output the results in the following JSON format:

[
    {{
        "statement": "direct quote of a key reasoning step from the chain of thought",
        "evidence": "direct quote of the specific text from the chain of thought that this statement is based on",
        "Verification": "true or false"
    }},
    ...
]
'''

# ICL Templates with demonstrations
QP_DEMON = '''The question is:

There are 7 candidates hired by Haier: F, G, H, I, W, X and Y.One of them needs to be assigned to the public relations department, three to the production department, and the other three to the sales department.The personnel allocation of these 7 employees must meet the following conditions: (1) H and Y must be allocated in the same department.(2) F and G cannot be allocated in the same department (3) If X is allocated in the sales department, W is allocated in the production department.(4) F must be allocated in the production department.\nWhich of the following lists is a complete and accurate list that cannot be assigned to the production department?\nA.F, I, X\nB.G, H, Y\nC.I, W\nD.G

The parsing result is:

[
    "One of them needs to be assigned to the public relations department, three to the production department, and the other three to the sales department",
    "H and Y must be allocated in the same department",
    "F and G cannot be allocated in the same department",
    "If X is allocated in the sales department, W is allocated in the production department",
    "F must be allocated in the production department"
]
'''

CP_DEMON = '''The question is:

In a magic show, from the seven magicians-G.H.K.L.N.P and Q, choose 6 people to play, and the performance is divided into two teams: 1 team and 2 teams.Each team consists of three positions: front, middle, and back.The magicians on the field happen to occupy one position each.The choice and location of the magician must meet the following conditions: (1) If G or H is arranged to play, they must be in the front.(2) If K is scheduled to play, he must be in the middle.(3) If L is scheduled to play, he must be on team 1.(4) Neither P nor K can be in the same team as N.(5) P cannot be in the same team as Q.(6) If H is in team 2, Q is in the middle of team 1.\nIf G is in team 1 and K is in team 2, which of the following magicians must be behind team 2?\nA.L\nB.N\nC.P\nD.Q

The conditions provided by the question is:

[
    "The magicians on the field happen to occupy one position each",
    "If G or H is arranged to play, they must be in the front",
    "If K is scheduled to play, he must be in the middle",
    "If L is scheduled to play, he must be on team 1",
    "Neither P nor K can be in the same team as N",
    "P cannot be in the same team as Q",
    "If H is in team 2, Q is in the middle of team 1",
    "G is in team 1 and K is in team 2"
]

The response to the question is:

Since G is in team 1, and K is in team 2, we know that L must be in team 1 (condition 3). We also know that P cannot be in the same team as N (condition 4), so P must be in team 2. Since H is in the front of team 1 (condition 1), and Q is in the middle of team 1 (condition 6), we can conclude that N must be in the back of team 1. Since L is in team 1, N must be behind L.

The parsing result of the response is:

[
    {
        "statement": "L must be in team 1",
        "evidence": "If L is scheduled to play, he must be on team 1",
        "Verification": "True"
    },
    {
        "statement": "P must be in team 2",
        "evidence": "G is in team 1, and K is in team 2, We also know that P cannot be in the same team as N",
        "Verification": "True"
    },
    {
        "statement": "N must be in the back of team 1",
        "evidence": "H is in the front of team 1 (condition 1), and Q is in the middle of team 1 (condition 6)",
        "Verification": "False"
    },
    {
        "statement": "N must be behind L",
        "evidence": "L is in team 1",
        "Verification": "False"
    }
]
'''

QP_ICL_TEMPLATE = '''Given a question, please parse and list all original fragments in the question that represent conditions related to solving the problem. Identify specific constraints, rules, or requirements that are essential for solving the problem.

Below is a demonstration, please refer to it:

{demon}

Now, the question you need to parse is: 

{question}

Output the list of parsing results in the format shown in the demonstration. Do not output other information.
'''

CP_ICL_TEMPLATE = '''Given a question, conditions in the question, and a chain of thought that attempts to solve the problem, your task is to analyze the reasoning by verifying whether each piece of evidence logically supports its statement.

Below is a demonstration showing the expected analysis approach:

{demon}

Now, analyze this problem with the SAME level of critical rigor:

The question is:
{question}

The conditions are:
{conditions}

The chain of thought is:
{cot}

Remember: Mark a statement as "true" ONLY if the evidence directly and logically supports it without requiring additional assumptions or logical leaps.

Output the results in exactly the same format as the demonstration, without additional explanation.'''

# System prompt
INFERENCE_SYSTEM_PROMPT = """You are a critical reasoner specializing in logical analysis of problem-solving attempts.

When analyzing chains of thought:
1. Focus EXCLUSIVELY on whether the cited evidence LOGICALLY SUPPORTS the statement
2. Apply strict standards of logical validity - the conclusion must follow directly from the premises
3. Be highly skeptical - mark statements as "false" unless the evidence DIRECTLY supports them
4. Look specifically for:
   - Conclusions that go beyond what the evidence establishes
   - Hidden assumptions not stated in the evidence
   - Logical jumps or gaps in reasoning
   - Misapplications of conditions from the problem

Your analysis should reflect the standards of formal logic, where a statement is only considered supported if it follows necessarily from the evidence provided.

Aim for a balanced assessment - in most reasoning chains, several statements will not be fully supported by their evidence."""

# QP QC System Prompt
QP_SYSTEM_PROMPT = """You are an expert at analyzing logical reasoning problems.

Your task is to verify and correct question parsing results, ensuring that:
1. All conditions are correctly extracted from the question
2. No important details or constraints are missing
3. No redundant or incorrect conditions are included
4. The conditions match the original wording in the question where appropriate

Be precise and thorough in your analysis. Focus on extracting the exact logical conditions as stated in the question."""

# CP QC System Prompt
CP_QC_SYSTEM_PROMPT = """You are an expert at analyzing logical reasoning in problem-solving scenarios.

Your task is to verify and correct chain-of-thought parsing results, ensuring that:
1. Statements are accurately extracted from the original chain of thought
2. Evidence is correctly identified from the chain of thought and conditions
3. Verification (True/False) accurately reflects whether each statement logically follows from the conditions

Be precise and thorough in your analysis. Distinguish between what is explicitly stated in the conditions and what is derived or inferred in the chain of thought."""

# QP QC template with examples
QP_QC_TEMPLATE_WITH_EXAMPLES = '''Given a question and the extracted conditions, please verify and correct any errors in the question parsing.

Here are some examples of good question parsing:
{demonstrations}

Now, please analyze the following case:

The question is:
{question}

The extracted conditions (question_parsing) are:
{question_parsing}

Verify the question parsing:
1. Check if all important conditions from the question are extracted correctly
2. Check if any extracted conditions are redundant or inaccurate
3. Check if the conditions match the original wording in the question

IMPORTANT: Ensure your response is valid JSON.

Provide your output as a JSON array of strings representing the corrected conditions:

[
    "condition 1",
    "condition 2",
    ...
]

Only output the JSON array with the corrected question parsing results, nothing else.'''

# CP QC template with examples
CP_QC_TEMPLATE_WITH_EXAMPLES = '''Given a question, the extracted conditions, and the chain-of-thought parsing results, please verify and correct any errors in the chain-of-thought parsing.

Here are some examples of good chain-of-thought parsing:
{demonstrations}

Now, please analyze the following case:

The question is:
{question}

The extracted conditions (question_parsing) are:
{question_parsing}

The chain of thought is:
{cot}

The CoT parsing results (cot_parsing) are:
{cot_parsing}

For each statement in the cot_parsing:
1. Verify if the statement is accurately extracted from the chain of thought
2. Verify if the evidence cited supports the statement and is found in either the chain of thought or the conditions
3. Verify if the verification (True/False) is correct according to the conditions provided in question_parsing

Pay special attention to:
- Whether the evidence actually supports the statement
- Whether the verification (True/False) is correct based on the conditions
- Whether the statement is actually found in or derivable from the chain of thought

IMPORTANT: Ensure your response is valid JSON. Use proper escaping for quotes within text. The "Verification" value must be exactly "True" or "False" (with quotes).

Provide a corrected version of the cot_parsing in exactly the same format:
[
    {{
        "statement": "statement 1",
        "evidence": "evidence 1",
        "Verification": "True or False"
    }},
    {{
        "statement": "statement 2",
        "evidence": "evidence 2", 
        "Verification": "True or False"
    }},
    ...
]

Only output the corrected cot_parsing as a valid JSON array, nothing else.'''

# Embedded examples for QP and CP QC
EMBEDDED_EXAMPLES = [
    {
        "question": "There are 7 outstanding students G, H, L, M, U, W and Z in a school.During the summer vacation, the school will send them to the United Kingdom and the United States for inspection.The school has only 7 students participating in this activity, and each person happens to go to one of these two countries.Considering the specialty of each student, this activity must meet the following conditions? (1) If G goes to the UK, then H To the United States.(2) If L goes to the UK, both M and U go to the US.(3) The country W went to was different from the country Z went to.(4) The country where U goes is different from the country where G goes.(5) If Z goes to the UK, then H also goes to the UK.\nIf G goes to the United States, which of the following must be true?\nA.H go to the UK\nB.L go to America\nC.M go to the UK\nD.W go to America",
        "question_parsing": [
            "The school has only 7 students participating in this activity, and each person happens to go to one of these two countries",
            "If G goes to the UK, then H To the United States",
            "If L goes to the UK, both M and U go to the US",
            "The country W went to was different from the country Z went to",
            "The country where U goes is different from the country where G goes",
            "If Z goes to the UK, then H also goes to the UK",
            "G goes to the United States"
        ],
        "answer": "b",
        "id": 162,
        "cot": "Since G goes to the United States, we need to analyze the conditions that follow. Condition (1) is not applicable since G is going to the US. Condition (2) is also not applicable since L's destination is not specified. Condition (3) does not provide any information about H, M, U, or W. Condition (4) states that U's destination is different from G's, which is the US, so U must go to the UK. Condition (5) is not applicable since Z's destination is not specified.",
        "cot_parsing": [
            {
                "statement": "Condition (1) is not applicable",
                "evidence": "G is going to the US",
                "Verification": "true"
            },
            {
                "statement": "Condition (2) is also not applicable",
                "evidence": "L's destination is not specified",
                "Verification": "true"
            },
            {
                "statement": "Condition (3) does not provide any information about H, M, U, or W",
                "evidence": "Condition (3)",
                "Verification": "false"
            },
            {
                "statement": "U must go to the UK",
                "evidence": "Condition (4) states that U's destination is different from G's, which is the US",
                "Verification": "true"
            },
            {
                "statement": "Condition (5) is not applicable",
                "evidence": "Z's destination is not specified",
                "Verification": "true"
            }
        ],
        "sel_idx": 92
    },
    {
        "question": "There are 6 clear flower porcelains: S, Y, M, Q, K, X.The production time of each piece is different.From left to right, the exhibitions are sorted in order from the earliest to the latest.The known sorting conditions are as follows: (1) M is older than X.(2) If Y is earlier than M, then Q is earlier than K and X.(3) If the age of M is earlier than Y, the age of K is earlier than Q and X.(4) The age of S is either earlier than Y or earlier than M, and both have neither.\nWhich of the following lists the possible exhibition order?\nA.Q.M.S, K.Y.X\nB.Q, K.Y, M.X, S\nC.Y.S, M.X.Q, K\nD.M, K, S, Q, Y, X",
        "question_parsing": [
            "From left to right, the exhibitions are sorted in order from the earliest to the latest",
            "M is older than X",
            "If Y is earlier than M, then Q is earlier than K and X",
            "If the age of M is earlier than Y, the age of K is earlier than Q and X",
            "The age of S is either earlier than Y or earlier than M, and both have neither"
        ],
        "answer": "d",
        "id": 374,
        "cot":"Let's analyze the given conditions:\n\n1. M is older than X.\n2. If Y is earlier than M, then Q is earlier than K and X.\n3. If M is earlier than Y, then K is earlier than Q and X.\n4. The age of S is either earlier than Y or earlier than M, and both have neither.\n\nFrom condition 1, we know M is older than X.\n\nFrom condition 2, if Y is earlier than M, then Q is earlier than K and X. This means that if Y is earlier than M, the order is Y, M, Q, K, X.\n\nFrom condition 3, if M is earlier than Y, then K is earlier than Q and X. This means that if M is earlier than Y, the order is M, K, Q, X.\n\nFrom condition 4, the age of S is either earlier than Y or earlier than M, and both have neither. This means that S can be placed anywhere in the sequence, but it cannot be placed between Y and M.",
        "cot_parsing": [
            {
                "statement":"if Y is earlier than M, the order is Y, M, Q, K, X",
                "evidence":"From condition 2, if Y is earlier than M, then Q is earlier than K and X",
                "Verification":"false"
            },
            {
                "statement":"if M is earlier than Y, the order is M, K, Q, X.",
                "evidence":"From condition 3, if M is earlier than Y, then K is earlier than Q and X",
                "Verification":"false"
            },
           {
                "statement":"S can be placed anywhere in the sequence, but it cannot be placed between Y and M",
                "evidence":"From condition 4, the age of S is either earlier than Y or earlier than M, and both have neither",
                "Verification":"false"
           }
        ],
        "sel_idx": 89
    },
    {
        "question": "In a magic show, from the seven magicians-G.H.K.L.N.P and Q, choose 6 people to play, and the performance is divided into two teams: 1 team and 2 teams.Each team consists of three positions: front, middle, and back.The magicians on the field happen to occupy one position each.The choice and location of the magician must meet the following conditions: (1) If G or H is arranged to play, they must be in the front.(2) If K is scheduled to play, he must be in the middle.(3) If L is scheduled to play, he must be on team 1.(4) Neither P nor K can be in the same team as N.(5) P cannot be in the same team as Q.(6) If H is in team 2, Q is in the middle of team 1.\nWhich of the following is an acceptable arrangement for the first team's performance?\nA.Before: Q, In: L, After: N\nB.Before: L, Middle: K, After: Q\nC.Before: Q, In: K, After: P\nD.Before: G, In: K, After: P",
        "question_parsing": [
            "In a magic show, from the seven magicians-G.H.K.L.N.P and Q, choose 6 people to play, and the performance is divided into two teams: 1 team and 2 teams",
            "Each team consists of three positions: front, middle, and back",
            "The magicians on the field happen to occupy one position each",
            "If G or H is arranged to play, they must be in the front",
            "If K is scheduled to play, he must be in the middle",
            "If L is scheduled to play, he must be on team 1",
            "Neither P nor K can be in the same team as N",
            "P cannot be in the same team as Q",
            "If H is in team 2, Q is in the middle of team 1"
        ],
        "answer": "a",
        "id": 476,
        "cot": "Let's analyze the conditions given:\n\n1. If G or H is arranged to play, they must be in the front.\n2. If K is scheduled to play, he must be in the middle.\n3. If L is scheduled to play, he must be on team 1.\n4. Neither P nor K can be in the same team as N.\n5. P cannot be in the same team as Q.\n6. If H is in team 2, Q is in the middle of team 1.\n\nWe need to find an acceptable arrangement for the first team's performance, which consists of three positions: front, middle, and back. Let's try to find a possible arrangement for the first team's performance. We can start by considering the conditions one by one.\n\nFrom condition 1, we know that if G or H is arranged to play, they must be in the front. Let's assume G is in the front.\n\nFrom condition 3, we know that if L is scheduled to play, he must be on team 1. Since L is not in the front, he must be in the middle or back. Let's assume L is in the middle.\n\nFrom condition 2, we know that if K is scheduled to play, he must be in the middle. Since L is already in the middle, K cannot be in the middle. Therefore, K must be in the back.\n\nFrom condition 4, we know that neither P nor K can be in the same team as N. Since K is in the back, P cannot be in the back. Therefore, P must be in the front or middle. However, P cannot be in the same team as Q, so P cannot be in the same team as L. Therefore, P must be in the front.\n\nFrom condition 6, we know that if H is in team 2, Q is in the middle of team 1. Since G is already in the front, H cannot be in team 1. Therefore, H must be in team 2.\n\nNow we have a possible arrangement for the first team's performance:\n\nFront: G, P\nMiddle: L\nBack: K\n\nPlease proceed to the next step",
        "cot_parsing": [
            {
                "statement": "he must be in the middle or back",
                "evidence": "From condition 1, we know that if G or H is arranged to play, they must be in the front. Let's assume G is in the front.\n\nFrom condition 3, we know that if L is scheduled to play, he must be on team 1. Since L is not in the front",
                "Verification": "false"
            },
            {
                "statement": "K must be in the back",
                "evidence": "From condition 2, we know that if K is scheduled to play, he must be in the middle. Since L is already in the middle, K cannot be in the middle.",
                "Verification": "false"
            },
            {
                "statement": "P cannot be in the back",
                "evidence": "From condition 4, we know that neither P nor K can be in the same team as N. Since K is in the back",
                "Verification": "false"
            },
            {
                "statement": "P must be in the front or middle",
                "evidence": "From condition 4, we know that neither P nor K can be in the same team as N. Since K is in the back, P cannot be in the back.",
                "Verification": "false"
            },
            {
                "statement": "P cannot be in the same team as L",
                "evidence": "P cannot be in the same team as Q",
                "Verification": "false"
            },
            {
                "statement": "P must be in the front",
                "evidence": "From condition 4, we know that neither P nor K can be in the same team as N. Since K is in the back, P cannot be in the back. Therefore, P must be in the front or middle. However, P cannot be in the same team as Q, so P cannot be in the same team as L.",
                "Verification": "false"
            },
            {
                "statement": "H must be in team 2",
                "evidence": "From condition 6, we know that if H is in team 2, Q is in the middle of team 1. Since G is already in the front, H cannot be in team 1.",
                "Verification": "false"
            }
        ],
        "sel_idx": 95
    },
    {
        "question": "Three Chinese students Zhang Lin, Zhao Qiang, Li Shan and three foreign students John, Jesse, and Anna went on a summer vacation.Alternative tourist destinations include Xi'an, Hangzhou, Dalian and Zhangjiajie.It is already known that: (1) each person can only go to one place; (2) wherever Chinese students go, foreign students must go; (3) where there are foreign students, Chinese students must go; (4) John went to Xi'an or Hangzhou, and Zhao Qiang went to Zhangjiajie.\nIf the conclusion of the question is true, which two people cannot go to Hangzhou at the same time?\nA.Zhang Lin and Li Shan\nB.Li Shan and Anna\nC.Jesse and Anna\nD.Zhang Lin and Jessie",
        "question_parsing": [
            "Three Chinese students Zhang Lin, Zhao Qiang, Li Shan and three foreign students John, Jesse, and Anna went on a summer vacation",
            "each person can only go to one place",
            "wherever Chinese students go, foreign students must go",
            "where there are foreign students, Chinese students must go",
            "John went to Xi'an or Hangzhou, and Zhao Qiang went to Zhangjiajie"
        ],
        "answer": "c",
        "id": 208,
        "cot": "Since it is known that (2) wherever Chinese students go, foreign students must go, and (3) where there are foreign students, Chinese students must go, we can conclude that if a Chinese student goes to a destination, a foreign student must also go to the same destination. We are also given that John went to Xi'an or Hangzhou, and Zhao Qiang went to Zhangjiajie. This means that if John went to Xi'an, Zhao Qiang must have gone to Zhangjiajie, and if John went to Hangzhou, Zhao Qiang must have gone to Zhangjiajie. Since Zhang Lin and Li Shan are Chinese students, they cannot both go to Hangzhou if John went to Hangzhou, because Zhao Qiang would have to go to Zhangjiajie, and there would be no foreign students left to go to Hangzhou. Similarly, if John went to Xi'an, Zhang Lin and Li Shan cannot both go to Hangzhou, because Zhao Qiang would have to go to Zhangjiajie, and there would be no Chinese students left to go to Hangzhou. Therefore, Zhang Lin and Li Shan cannot go to Hangzhou at the same time.",
        "cot_parsing": [
            {
                "statement": "if a Chinese student goes to a destination, a foreign student must also go to the same destination",
                "evidence": "(2) wherever Chinese students go, foreign students must go, and (3) where there are foreign students, Chinese students must go",
                "Verification": "true"
            },
            {
                "statement": "if John went to Xi'an, Zhao Qiang must have gone to Zhangjiajie, and if John went to Hangzhou, Zhao Qiang must have gone to Zhangjiajie",
                "evidence": "John went to Xi'an or Hangzhou, and Zhao Qiang went to Zhangjiajie.",
                "Verification": "false"
            },
            {
                "statement": "Zhang Lin and Li Shan cannot both go to Hangzhou if John went to Hangzhou",
                "evidence": "Zhang Lin and Li Shan are Chinese students, Zhao Qiang would have to go to Zhangjiajie, and there would be no foreign students left to go to Hangzhou",
                "Verification": "false"
            },
            {
                "statement": "if John went to Xi'an, Zhang Lin and Li Shan cannot both go to Hangzhou",
                "evidence": "Zhao Qiang would have to go to Zhangjiajie, and there would be no Chinese students left to go to Hangzhou",
                "Verification": "false"
            }
        ],
        "sel_idx": 97
    },
    {
        "question": "Seven people from G, H, J, K, L, M and O traveled together.In their 4-day camping life, two people are scheduled to cook for everyone every day.It is required that the four-day staff arrangement cannot be repeated, and there is exactly one person on duty for two days.In addition, the duty arrangement must also meet the following conditions? (1) J should be placed on the next day of duty H.(2) The person on duty for two days must be on the fourth day, not the third day.(3) G must be on the same day as one of J and O.(4) Arrange K at least one day on the first or second day.(5) O must be on the third day.\nWhich of the following people can't be on duty for two days?\nA.G\nB.H\nC.K\nD.L",
        "question_parsing": [
            "In their 4-day camping life, two people are scheduled to cook for everyone every day",
            "It is required that the four-day staff arrangement cannot be repeated",
            "there is exactly one person on duty for two days",
            "J should be placed on the next day of duty H", 
            "The person on duty for two days must be on the fourth day, not the third day",
            "G must be on the same day as one of J and O",
            "Arrange K at least one day on the first or second day",
            "O must be on the third day."
        ],
        "answer": "a",
        "id": 54,
        "cot": "Let's analyze the given conditions. We know that the duty arrangement cannot be repeated, and there is exactly one person on duty for two days. We also know that J should be placed on the next day of duty H, and G must be on the same day as one of J and O. Additionally, O must be on the third day, and K must be on at least one day on the first or second day.\n\nFrom condition (2), we know that the person on duty for two days must be on the fourth day, not the third day. This means that the person on duty for two days cannot be O, since O is already on the third day.\n\nFrom condition (3), we know that G must be on the same day as one of J and O. Since O is on the third day, G must be on the same day as J. This means that G and J are on the same day, and G is not on duty for two days.",
        "cot_parsing": [
            {
                "statement":"the person on duty for two days cannot be O, ",
                "evidence":"The person on duty for two days must be on the fourth day, not the third day. since O is already on the third day.",
                "Verification":"true"
            },
            {
                "statement":"G must be on the same day as J.",
                "evidence":"O is on the third day, G must be on the same day as one of J and O.",
                "Verification":"false"
            },
           {
                "statement":"G and J are on the same day, and G is not on duty for two days.",
                "evidence":"O is on the third day, G must be on the same day as J.",
                "Verification":"false"
           }
        ],
        "sel_idx": 84
    }]
# Extraction functions
def extract_question_parsing(text):
    """Extract question parsing from model output"""
    try:
        # Try to find a JSON array (list) in the text
        list_pattern = r'\[\s*\"[^\"]+(?:\",\s*\"[^\"]+\")*\"\s*\]'
        match = re.search(list_pattern, text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        
        # Alternatively, try to extract lines that look like list items
        list_items = re.findall(r'"([^"]+)"', text)
        if list_items:
            return list_items
        
        return []
    except Exception as e:
        print(f"Error extracting question parsing: {e}")
        return []

def extract_cot_parsing(text):
    """Extract CoT parsing from model output"""
    try:
        # Find a JSON array of objects in the text
        json_pattern = r'\[\s*\{(?:\s*\"[^\"]+\"\s*:\s*\"[^\"]+\"\s*,?)+\}(?:\s*,\s*\{(?:\s*\"[^\"]+\"\s*:\s*\"[^\"]+\"\s*,?)+\})*\s*\]'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed_data = json.loads(json_str)
            
            # Normalize the verification field for consistency
            for item in parsed_data:
                if "Verification" in item:
                    # Convert to lowercase and then capitalize first letter for consistency
                    verification = item["Verification"].lower()
                    if verification in ["true", "correct"]:
                        item["Verification"] = "True"
                    elif verification in ["false", "incorrect"]:
                        item["Verification"] = "False"
            
            return parsed_data
        return []
    except Exception as e:
        print(f"Error extracting cot parsing: {e}")
        return []

def extract_cp_qc_results(text):
    """Extract the corrected cot_parsing from the QC output with improved error handling"""
    try:
        # First attempt: Find a JSON array of objects in the text
        json_pattern = r'\[\s*\{(?:\s*\"[^\"]+\"\s*:\s*\"[^\"]+\"\s*,?)+\}(?:\s*,\s*\{(?:\s*\"[^\"]+\"\s*:\s*\"[^\"]+\"\s*,?)+\})*\s*\]'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        
        # Second attempt: If the pattern didn't match, try finding brackets and parsing
        if text.strip().startswith('[') and text.strip().endswith(']'):
            try:
                return json.loads(text.strip())
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                # Attempt to fix common JSON syntax issues
                fixed_text = text.strip()
                fixed_text = re.sub(r'(?<!\")True(?!\")', '"True"', fixed_text)  # Fix unquoted True
                fixed_text = re.sub(r'(?<!\")False(?!\")', '"False"', fixed_text)  # Fix unquoted False
                fixed_text = re.sub(r',\s*}', '}', fixed_text)  # Remove trailing commas
                fixed_text = re.sub(r',\s*]', ']', fixed_text)  # Remove trailing commas
                try:
                    return json.loads(fixed_text)
                except json.JSONDecodeError:
                    # If that didn't work, print problematic part for debugging
                    print(f"Error location around: {text[max(0, e.pos-30):min(len(text), e.pos+30)]}")
                    return []
        
        # If all else fails, try a more manual approach to extract individual objects
        objects = []
        object_pattern = r'\{\s*\"statement\":\s*\"[^\"]+\",\s*\"evidence\":\s*\"[^\"]+\",\s*\"Verification\":\s*\"(?:True|False)\"\s*\}'
        for match in re.finditer(object_pattern, text):
            try:
                obj = json.loads(match.group(0))
                objects.append(obj)
            except:
                pass
        
        if objects:
            return objects
            
        return []
    except Exception as e:
        print(f"Error extracting CP QC results: {e}")
        # Print a snippet of the problematic text for debugging
        if isinstance(text, str) and len(text) > 0:
            print(f"Text snippet: {text[:min(100, len(text))]}")
        return []

def extract_qp_qc_results(text):
    """Extract the corrected question_parsing from the QC output with improved error handling"""
    try:
        # First attempt: Try to find a JSON array in the text
        json_pattern = r'\[\s*\"[^\"]+(?:\",\s*\"[^\"]+\")*\"\s*\]'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        
        # Second attempt: If the pattern didn't match, try finding brackets and parsing
        if text.strip().startswith('[') and text.strip().endswith(']'):
            try:
                return json.loads(text.strip())
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                # Print problematic part for debugging
                print(f"Error location around: {text[max(0, e.pos-30):min(len(text), e.pos+30)]}")
                
                # Try to clean up common issues
                cleaned_text = text.strip()
                cleaned_text = re.sub(r',\s*]', ']', cleaned_text)  # Remove trailing commas
                try:
                    return json.loads(cleaned_text)
                except json.JSONDecodeError:
                    return []
        
        # Third attempt: Try to find quoted strings and manually build the array
        condition_pattern = r'"([^"]+)"'
        conditions = []
        for match in re.finditer(condition_pattern, text):
            conditions.append(match.group(1))
        
        if conditions:
            return conditions
            
        return []
    except Exception as e:
        print(f"Error extracting QP QC results: {e}")
        # Print a snippet of the problematic text for debugging
        if isinstance(text, str) and len(text) > 0:
            print(f"Text snippet: {text[:min(100, len(text))]}")
        return []

def create_demonstrations(all_examples, current_sample_id, num_examples=2, use_embedded=True):
    """Create demonstrations from examples for few-shot learning
    
    Args:
        all_examples: List of examples to choose from (if use_embedded=False)
        current_sample_id: ID of the current sample (to avoid using it as an example)
        num_examples: Number of examples to include
        use_embedded: Whether to use embedded examples instead of from file
        
    Returns:
        String containing formatted demonstrations
    """
    if use_embedded:
        # Use the embedded examples
        selected_examples = random.sample(EMBEDDED_EXAMPLES, min(num_examples, len(EMBEDDED_EXAMPLES)))
    else:
        # Filter out the current sample (if it exists in examples) to avoid using it as its own example
        filtered_examples = [ex for ex in all_examples if ex.get('id') != current_sample_id]
        
        # Select random examples
        selected_examples = random.sample(filtered_examples, min(num_examples, len(filtered_examples)))
    
    demonstrations = ""
    for i, example in enumerate(selected_examples):
        demonstrations += f"\nExample {i+1}:\n"
        demonstrations += f"Question: {example['question']}\n\n"
        demonstrations += f"Extracted conditions:\n{json.dumps(example['question_parsing'], indent=2)}\n\n"
        
        if 'cot' in example and 'cot_parsing' in example:
            demonstrations += f"Chain of thought:\n{example['cot']}\n\n"
            demonstrations += f"Original CoT parsing:\n{json.dumps(example['cot_parsing'], indent=2)}\n\n"
            demonstrations += f"These CoT parsing results correctly extract statements from the chain of thought, identify evidence, and verify each statement according to the conditions.\n"
        else:
            demonstrations += f"These question parsing results correctly extract all important conditions from the question.\n"
        
    return demonstrations

def process_all_tasks(input_file, output_prefix, examples_file=None, save_every=10, use_icl=True, 
                      num_examples=2, debug=False, use_embedded_examples=True):
    """
    Run the complete pipeline: inference -> CP QC -> QP QC
    
    Args:
        input_file: Path to input JSON file
        output_prefix: Prefix for output file names
        examples_file: Path to examples JSON file for few-shot learning (optional if use_embedded_examples=True)
        save_every: How often to save partial results (number of samples)
        use_icl: Whether to use in-context learning templates
        num_examples: Number of examples to include in the QC prompts
        debug: Whether to print debug information
        use_embedded_examples: Whether to use embedded examples instead of loading from file
    """
    # Define output file paths
    inference_output = f"{output_prefix}_inference.json"
    cp_qc_output = f"{output_prefix}_cp_qc.json"
    final_output = f"{output_prefix}_final.json"
    
    # Load model once for all tasks
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="meta-llama/meta-Llama-3-8B-Instruct",
        max_seq_length=4096*2,
        load_in_4bit=False,
        fast_inference=True,
        gpu_memory_utilization=0.9,
    )
    
    # Load all examples for few-shot learning if needed
    all_examples = []
    if not use_embedded_examples and examples_file:
        print(f"Loading examples from {examples_file}...")
        with open(examples_file, 'r', encoding='utf-8') as f:
            all_examples = json.load(f)
        print(f"Loaded {len(all_examples)} examples")
    elif not use_embedded_examples:
        print("Warning: examples_file not provided but use_embedded_examples is False")
    else:
        print(f"Using {len(EMBEDDED_EXAMPLES)} embedded examples")
    
    # Seed random for reproducibility
    random.seed(42)
    
    # Load the original test data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Step 1: Run inference (Question Parsing)
    print(f"Step 1.1: Running Question Parsing...")
    question_parsing_count = 0
    for i, sample in enumerate(tqdm(test_data, desc="Question Parsing")):
        question = sample["question"]
        
        # Format QP prompt
        if use_icl:
            qp_prompt = QP_ICL_TEMPLATE.format(
                demon=QP_DEMON, 
                question=question
            )
        else:
            qp_prompt = QP_TEMPLATE.format(
                question=question
            )
        
        qp_formatted = tokenizer.apply_chat_template([
            {"role": "system", "content": INFERENCE_SYSTEM_PROMPT},
            {"role": "user", "content": qp_prompt},
        ], tokenize=False, add_generation_prompt=True)
        
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=2048,
        )
        
        qp_output = model.fast_generate(
            qp_formatted,
            sampling_params=sampling_params,
        )[0].outputs[0].text
        
        question_parsing = extract_question_parsing(qp_output)
        
        # Retry if question parsing failed
        retry_count = 0
        max_retries = 5
        while not question_parsing and retry_count < max_retries:
            print(f"\nEmpty question_parsing for sample {i}. Retrying ({retry_count+1}/{max_retries})...")
            time.sleep(1)
            
            # Increase temperature for retry
            retry_temp = 0.1 + (retry_count * 0.1)
            
            sampling_params = SamplingParams(
                temperature=retry_temp,
                top_p=0.95,
                max_tokens=2048,
            )
            
            qp_output = model.fast_generate(
                qp_formatted,
                sampling_params=sampling_params,
            )[0].outputs[0].text
            
            question_parsing = extract_question_parsing(qp_output)
            retry_count += 1
        
        # Update sample with the parsed question conditions
        if question_parsing:
            sample["question_parsing"] = question_parsing
            question_parsing_count += 1
        
        # Save intermediate results periodically
        if (i + 1) % save_every == 0:
            with open(f"{inference_output}.partial", 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=4, ensure_ascii=True)
            print(f"\nSaved partial results ({i+1}/{len(test_data)} samples processed)")
    
    # Save results after Question Parsing
    with open(inference_output, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=True)
    print(f"Question Parsing completed. Successful parsing: {question_parsing_count}/{len(test_data)}. Results saved to {inference_output}")
    
    # Step 1.2: Run inference (CoT Parsing)
    print(f"Step 1.2: Running Chain of Thought Parsing...")
    cot_parsing_count = 0
    for i, sample in enumerate(tqdm(test_data, desc="CoT Parsing")):
        # Only process samples that have both question_parsing and cot
        if "question_parsing" in sample and sample.get("cot"):
            question = sample["question"]
            conditions = json.dumps(sample["question_parsing"], indent=2)
            cot = sample["cot"]
            
            # Format CP prompt
            if use_icl:
                cp_prompt = CP_ICL_TEMPLATE.format(
                    demon=CP_DEMON,
                    question=question,
                    conditions=conditions,
                    cot=cot
                )
            else:
                cp_prompt = CP_TEMPLATE.format(
                    question=question,
                    conditions=conditions,
                    cot=cot
                )
            
            cp_formatted = tokenizer.apply_chat_template([
                {"role": "system", "content": INFERENCE_SYSTEM_PROMPT},
                {"role": "user", "content": cp_prompt},
            ], tokenize=False, add_generation_prompt=True)
            
            sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.95,
                max_tokens=2048,
            )
            
            cp_output = model.fast_generate(
                cp_formatted,
                sampling_params=sampling_params,
            )[0].outputs[0].text
            
            cot_parsing = extract_cot_parsing(cp_output)
            
            # Retry if cot parsing failed
            retry_count = 0
            max_retries = 5
            while not cot_parsing and retry_count < max_retries:
                print(f"\nEmpty cot_parsing for sample {i}. Retrying ({retry_count+1}/{max_retries})...")
                time.sleep(1)
                
                # Increase temperature for retry
                retry_temp = 0.1 + (retry_count * 0.1)
                
                sampling_params = SamplingParams(
                    temperature=retry_temp,
                    top_p=0.95,
                    max_tokens=2048,
                )
                
                cp_output = model.fast_generate(
                    cp_formatted,
                    sampling_params=sampling_params,
                )[0].outputs[0].text
                
                cot_parsing = extract_cot_parsing(cp_output)
                retry_count += 1
            
            # Update sample with the parsed CoT
            if cot_parsing:
                sample["cot_parsing"] = cot_parsing
                cot_parsing_count += 1
        
        # Save intermediate results periodically
        if (i + 1) % save_every == 0:
            with open(f"{inference_output}.partial", 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=4, ensure_ascii=True)
            print(f"\nSaved partial results ({i+1}/{len(test_data)} samples processed)")
    
    # Save results after CoT Parsing
    with open(inference_output, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=True)
    print(f"CoT Parsing completed. Successful parsing: {cot_parsing_count}/{len(test_data)}. Results saved to {inference_output}")
    
    # Step 2: Run CP QC
    print(f"Step 2: Running CP QC, saving to {cp_qc_output}")
    cp_qc_changes_count = 0
    for i, sample in enumerate(tqdm(test_data, desc="CP QC")):
        # Only run QC if we have the necessary components
        if (sample.get("cot") and 
            sample.get("question_parsing") and 
            sample.get("cot_parsing")):
            
            # Generate random demonstrations for this sample
            current_sample_id = sample.get('id', i)
            demonstrations = create_demonstrations(all_examples, current_sample_id, num_examples, use_embedded=use_embedded_examples)
            
            cp_qc_prompt = CP_QC_TEMPLATE_WITH_EXAMPLES.format(
                demonstrations=demonstrations,
                question=sample["question"],
                question_parsing=json.dumps(sample["question_parsing"], indent=2),
                cot=sample["cot"],
                cot_parsing=json.dumps(sample["cot_parsing"], indent=2)
            )
            
            cp_qc_formatted = tokenizer.apply_chat_template([
                {"role": "system", "content": CP_QC_SYSTEM_PROMPT},
                {"role": "user", "content": cp_qc_prompt},
            ], tokenize=False, add_generation_prompt=True)
            
            sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.95,
                max_tokens=2048,
            )
            
            # Run QC inference
            cp_qc_out = model.fast_generate(
                cp_qc_formatted,
                sampling_params=sampling_params,
            )[0].outputs[0].text
            
            corrected_cp = extract_cp_qc_results(cp_qc_out)
            
            # Retry if QC parsing failed
            retry_count = 0
            max_retries = 5
            while not corrected_cp and retry_count < max_retries:
                print(f"\nEmpty corrected_cp for sample {current_sample_id}. Retrying ({retry_count+1}/{max_retries})...")
                time.sleep(1)
                
                # Try with different random examples
                demonstrations = create_demonstrations(all_examples, current_sample_id, num_examples, use_embedded=use_embedded_examples)
                
                cp_qc_prompt = CP_QC_TEMPLATE_WITH_EXAMPLES.format(
                    demonstrations=demonstrations,
                    question=sample["question"],
                    question_parsing=json.dumps(sample["question_parsing"], indent=2),
                    cot=sample["cot"],
                    cot_parsing=json.dumps(sample["cot_parsing"], indent=2)
                )
                
                cp_qc_formatted = tokenizer.apply_chat_template([
                    {"role": "system", "content": CP_QC_SYSTEM_PROMPT},
                    {"role": "user", "content": cp_qc_prompt},
                ], tokenize=False, add_generation_prompt=True)
                
                # Increase temperature for retry
                retry_temp = 0.1 + (retry_count * 0.1)
                
                sampling_params = SamplingParams(
                    temperature=retry_temp,
                    top_p=0.95,
                    max_tokens=2048,
                )
                
                cp_qc_out = model.fast_generate(
                    cp_qc_formatted,
                    sampling_params=sampling_params,
                )[0].outputs[0].text
                
                corrected_cp = extract_cp_qc_results(cp_qc_out)
                retry_count += 1
            
            # Update the cot_parsing if corrections were found
            if corrected_cp:
                original_cp_json = json.dumps(sample["cot_parsing"], sort_keys=True)
                corrected_cp_json = json.dumps(corrected_cp, sort_keys=True)
                
                if original_cp_json != corrected_cp_json:
                    sample["cot_parsing"] = corrected_cp
                    cp_qc_changes_count += 1
                    print(f"\nCP QC changes applied to sample {current_sample_id}")
        
        # Save intermediate results periodically
        if (i + 1) % save_every == 0:
            with open(f"{cp_qc_output}.partial", 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=4, ensure_ascii=True)
            print(f"\nSaved partial results ({i+1}/{len(test_data)} samples processed)")
    
    # Save results after CP QC
    with open(cp_qc_output, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=True)
    print(f"CP QC completed. Changes applied to {cp_qc_changes_count}/{len(test_data)} samples. Results saved to {cp_qc_output}")
    
    # Step 3: Run QP QC
    print(f"Step 3: Running QP QC, saving to {final_output}")
    qp_qc_changes_count = 0
    for i, sample in enumerate(tqdm(test_data, desc="QP QC")):
        # Only run QC if we have the necessary components
        if sample.get("question") and sample.get("question_parsing"):
            # Generate random demonstrations for this sample
            current_sample_id = sample.get('id', i)
            demonstrations = create_demonstrations(all_examples, current_sample_id, num_examples, use_embedded=use_embedded_examples)
            
            qp_qc_prompt = QP_QC_TEMPLATE_WITH_EXAMPLES.format(
                demonstrations=demonstrations,
                question=sample["question"],
                question_parsing=json.dumps(sample["question_parsing"], indent=2)
            )
            
            qp_qc_formatted = tokenizer.apply_chat_template([
                {"role": "system", "content": QP_SYSTEM_PROMPT},
                {"role": "user", "content": qp_qc_prompt},
            ], tokenize=False, add_generation_prompt=True)
            
            sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.95,
                max_tokens=2048,
            )
            
            # Run QC inference
            qp_qc_out = model.fast_generate(
                qp_qc_formatted,
                sampling_params=sampling_params,
            )[0].outputs[0].text
            
            if debug:
                print(f"\nDEBUG - Raw model output for sample {current_sample_id}:")
                print(qp_qc_out[:200] + "..." if len(qp_qc_out) > 200 else qp_qc_out)
            
            corrected_qp = extract_qp_qc_results(qp_qc_out)
            
            # Retry if QC parsing failed
            retry_count = 0
            max_retries = 5
            while not corrected_qp and retry_count < max_retries:
                print(f"\nEmpty corrected_qp for sample {current_sample_id}. Retrying ({retry_count+1}/{max_retries})...")
                time.sleep(1)
                
                # Try with different random examples
                demonstrations = create_demonstrations(all_examples, current_sample_id, num_examples, use_embedded=use_embedded_examples)
                
                qp_qc_prompt = QP_QC_TEMPLATE_WITH_EXAMPLES.format(
                    demonstrations=demonstrations,
                    question=sample["question"],
                    question_parsing=json.dumps(sample["question_parsing"], indent=2)
                )
                
                qp_qc_formatted = tokenizer.apply_chat_template([
                    {"role": "system", "content": QP_SYSTEM_PROMPT},
                    {"role": "user", "content": qp_qc_prompt},
                ], tokenize=False, add_generation_prompt=True)
                
                # Increase temperature for retry
                retry_temp = 0.1 + (retry_count * 0.1)
                
                sampling_params = SamplingParams(
                    temperature=retry_temp,
                    top_p=0.95,
                    max_tokens=2048,
                )
                
                qp_qc_out = model.fast_generate(
                    qp_qc_formatted,
                    sampling_params=sampling_params,
                )[0].outputs[0].text
                
                if debug:
                    print(f"\nDEBUG - Retry {retry_count+1} raw output:")
                    print(qp_qc_out[:200] + "..." if len(qp_qc_out) > 200 else qp_qc_out)
                
                corrected_qp = extract_qp_qc_results(qp_qc_out)
                retry_count += 1
                
            # Basic validation of the corrected QP
            valid_qp = True
            if corrected_qp:
                # Check that all items are strings
                if not all(isinstance(item, str) for item in corrected_qp):
                    print(f"Invalid QP format for sample {current_sample_id}: not all items are strings")
                    valid_qp = False
                    
                # Check that we have at least one condition
                if len(corrected_qp) == 0:
                    print(f"Invalid QP format for sample {current_sample_id}: empty conditions list")
                    valid_qp = False
            else:
                valid_qp = False
            
            # Update the question_parsing if valid corrections were found
            if valid_qp:
                original_qp_json = json.dumps(sample["question_parsing"], sort_keys=True)
                corrected_qp_json = json.dumps(corrected_qp, sort_keys=True)
                
                if original_qp_json != corrected_qp_json:
                    sample["question_parsing"] = corrected_qp
                    qp_qc_changes_count += 1
                    print(f"\nQP QC changes applied to sample {current_sample_id}")
                    
                    if debug:
                        print("Original:", original_qp_json)
                        print("Corrected:", corrected_qp_json)
        
        # Save intermediate results periodically
        if (i + 1) % save_every == 0:
            with open(f"{final_output}.partial", 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=4, ensure_ascii=True)
            print(f"\nSaved partial results ({i+1}/{len(test_data)} samples processed)")
    
    # Save final results
    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=True)
    
    print("\nComplete pipeline has finished successfully.")
    print(f"Final results saved to {final_output}")
    print(f"QP parsing successful: {question_parsing_count}/{len(test_data)} samples")
    print(f"CoT parsing successful: {cot_parsing_count}/{len(test_data)} samples")
    print(f"CP QC changes applied to: {cp_qc_changes_count}/{len(test_data)} samples")
    print(f"QP QC changes applied to: {qp_qc_changes_count}/{len(test_data)} samples")
    
    return test_data

def main():
    """Main function to parse arguments and run the pipeline"""
    parser = argparse.ArgumentParser(description="Enhanced inference for LLM Self-Reasoning")
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input JSON file')
    parser.add_argument('--output_prefix', type=str, default='results',
                        help='Prefix for output file names')
    parser.add_argument('--save_every', type=int, default=10,
                        help='How often to save partial results (number of samples)')
    parser.add_argument('--use_icl', action='store_true', default=True,
                        help='Whether to use in-context learning templates')
    parser.add_argument('--num_examples', type=int, default=2,
                        help='Number of examples to include in the QC prompts')
    parser.add_argument('--debug', action='store_true',
                        help='Whether to print debug information')
    
    args = parser.parse_args()
    
    # Always run all tasks in sequence
    process_all_tasks(
        input_file=args.input_file,
        output_prefix=args.output_prefix,
        save_every=args.save_every,
        use_icl=args.use_icl,
        num_examples=args.num_examples,
        debug=args.debug,
        use_embedded_examples=True
    )

if __name__ == "__main__":
    main()