# the api is http://127.0.0.1:5000/api/v1/generate
# we need to send a post request to this api with the following json data:
# {
#     "prompt": "The quick brown fox jumps over the lazy dog.",
# }

import datetime
import requests


def gather_contexts(max_char=1000):
    with open('examples/documents.txt') as f:
        context = f.read()

    # cut up the text in maximum max_char characters
    # preferably at a full stop \n\n\n
    # second best is interline \n\n
    # third best is \n

    contexts = []

    length = 0

    while len(context) > max_char:
        # find the last full stop
        index = context.rfind('\n\n\n', 0, max_char)
        if index <= 0:
            index = context.rfind('\n\n', 0, max_char)
            if index <= 0:
                index = context.rfind('\n', 0, max_char)
                if index <= 0:
                    index = max_char

        contexts.append(context[:index])
        context = context[index:]

        length += index

        if len(context) < max_char:
            contexts.append(context)
            break

    return contexts


def build_prompt(context, prompt):
    # inside is a {context} placeholder
    prompt = prompt.replace('{context}', context)

    return prompt


def generate(prompt):
    url = 'http://127.0.0.1:5000/api/v1/generate'
    data = {
        "prompt": prompt,
        "negative prompt": "context, input, ça, it, ce, cela, ceci, this, these",
        "top_k": 1,
        "temperature": 0.3,
        "min_length": 50,
        "max_new_tokens": 1000,
        "guidance_scale": 2
    }
    response = requests.post(url,
                             json=data,
                             headers={'Content-Type': 'application/json'},
                             timeout=500)

    return response.json()['results'][0]['text']


if __name__ == '__main__':
    contexts = gather_contexts()

    with open('prompts/resume.txt') as f:
        prompt_base = f.read()

    # timestamp for differentiating files
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    with open(f'examples/private/resume_{timestamp}.txt', mode="w") as f:
        for context in contexts:
            for repeat in range(4):
                prompt = build_prompt(context, prompt_base)
                answer = generate(prompt)

                print(f"The prompt for {repeat} is: {prompt} {answer}")

                # write to file
                f.write(f"Q:{answer}")
                f.write('\n\n')
