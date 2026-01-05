Migrate to the Responses API

============================



The \[Responses API](/docs/api-reference/responses) is our new API primitive, an evolution of \[Chat Completions](/docs/api-reference/chat) which brings added simplicity and powerful agentic primitives to your integrations.



\*\*While Chat Completions remains supported, Responses is recommended for all new projects.\*\*



About the Responses API

-----------------------



The Responses API is a unified interface for building powerful, agent-like applications. It contains:



\*   Built-in tools like \[web search](/docs/guides/tools-web-search), \[file search](/docs/guides/tools-file-search) , \[computer use](/docs/guides/tools-computer-use), \[code interpreter](/docs/guides/tools-code-interpreter), and \[remote MCPs](/docs/guides/tools-remote-mcp).

\*   Seamless multi-turn interactions that allow you to pass previous responses for higher accuracy reasoning results.

\*   Native multimodal support for text and images.



Responses benefits

------------------



The Responses API contains several benefits over Chat Completions:



\*   \*\*Better performance\*\*: Using reasoning models, like GPT-5, with Responses will result in better model intelligence when compared to Chat Completions. Our internal evals reveal a 3% improvement in SWE-bench with same prompt and setup.

\*   \*\*Agentic by default\*\*: The Responses API is an agentic loop, allowing the model to call multiple tools, like `web\_search`, `image\_generation`, `file\_search`, `code\_interpreter`, remote MCP servers, as well as your own custom functions, within the span of one API request.

\*   \*\*Lower costs\*\*: Results in lower costs due to improved cache utilization (40% to 80% improvement when compared to Chat Completions in internal tests).

\*   \*\*Stateful context\*\*: Use `store: true` to maintain state from turn to turn, preserving reasoning and tool context from turn-to-turn.

\*   \*\*Flexible inputs\*\*: Pass a string with input or a list of messages; use instructions for system-level guidance.

\*   \*\*Encrypted reasoning\*\*: Opt-out of statefulness while still benefiting from advanced reasoning.

\*   \*\*Future-proof\*\*: Future-proofed for upcoming models.



|Capabilities|Chat Completions API|Responses API|

|---|---|---|

|Text generation|||

|Audio||Coming soon|

|Vision|||

|Structured Outputs|||

|Function calling|||

|Web search|||

|File search|||

|Computer use|||

|Code interpreter|||

|MCP|||

|Image generation|||

|Reasoning summaries|||



\### Examples



See how the Responses API compares to the Chat Completions API in specific scenarios.



\#### Messages vs. Items



Both APIs make it easy to generate output from our models. The input to, and result of, a call to Chat completions is an array of \_Messages\_, while the Responses API uses \_Items\_. An Item is a union of many types, representing the range of possibilities of model actions. A `message` is a type of Item, as is a `function\_call` or `function\_call\_output`. Unlike a Chat Completions Message, where many concerns are glued together into one object, Items are distinct from one another and better represent the basic unit of model context.



Additionally, Chat Completions can return multiple parallel generations as `choices`, using the `n` param. In Responses, we've removed this param, leaving only one generation.



Chat Completions API



```

from openai import OpenAI

client = OpenAI()



completion = client.chat.completions.create(

&nbsp; model="gpt-5",

&nbsp; messages=\[

&nbsp;     {

&nbsp;         "role": "user",

&nbsp;         "content": "Write a one-sentence bedtime story about a unicorn."

&nbsp;     }

&nbsp; ]

)



print(completion.choices\[0].message.content)

```



Responses API



```

from openai import OpenAI

client = OpenAI()



response = client.responses.create(

&nbsp; model="gpt-5",

&nbsp; input="Write a one-sentence bedtime story about a unicorn."

)



print(response.output\_text)

```



When you get a response back from the Responses API, the fields differ slightly. Instead of a `message`, you receive a typed `response` object with its own `id`. Responses are stored by default. Chat completions are stored by default for new accounts. To disable storage when using either API, set `store: false`.



The objects you recieve back from these APIs will differ slightly. In Chat Completions, you receive an array of `choices`, each containing a `message`. In Responses, you receive an array of Items labled `output`.



Chat Completions API



```

{

&nbsp; "id": "chatcmpl-C9EDpkjH60VPPIB86j2zIhiR8kWiC",

&nbsp; "object": "chat.completion",

&nbsp; "created": 1756315657,

&nbsp; "model": "gpt-5-2025-08-07",

&nbsp; "choices": \[

&nbsp;   {

&nbsp;     "index": 0,

&nbsp;     "message": {

&nbsp;       "role": "assistant",

&nbsp;       "content": "Under a blanket of starlight, a sleepy unicorn tiptoed through moonlit meadows, gathering dreams like dew to tuck beneath its silver mane until morning.",

&nbsp;       "refusal": null,

&nbsp;       "annotations": \[]

&nbsp;     },

&nbsp;     "finish\_reason": "stop"

&nbsp;   }

&nbsp; ],

&nbsp; ...

}

```



Responses API



```

{

&nbsp; "id": "resp\_68af4030592c81938ec0a5fbab4a3e9f05438e46b5f69a3b",

&nbsp; "object": "response",

&nbsp; "created\_at": 1756315696,

&nbsp; "model": "gpt-5-2025-08-07",

&nbsp; "output": \[

&nbsp;   {

&nbsp;     "id": "rs\_68af4030baa48193b0b43b4c2a176a1a05438e46b5f69a3b",

&nbsp;     "type": "reasoning",

&nbsp;     "content": \[],

&nbsp;     "summary": \[]

&nbsp;   },

&nbsp;   {

&nbsp;     "id": "msg\_68af40337e58819392e935fb404414d005438e46b5f69a3b",

&nbsp;     "type": "message",

&nbsp;     "status": "completed",

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "type": "output\_text",

&nbsp;         "annotations": \[],

&nbsp;         "logprobs": \[],

&nbsp;         "text": "Under a quilt of moonlight, a drowsy unicorn wandered through quiet meadows, brushing blossoms with her glowing horn so they sighed soft lullabies that carried every dreamer gently to sleep."

&nbsp;       }

&nbsp;     ],

&nbsp;     "role": "assistant"

&nbsp;   }

&nbsp; ],

&nbsp; ...

}

```



\### Additional differences



\*   Responses are stored by default. Chat completions are stored by default for new accounts. To disable storage in either API, set `store: false`.

\*   \[Reasoning](/docs/guides/reasoning) models have a richer experience in the Responses API with \[improved tool usage](/docs/guides/reasoning#keeping-reasoning-items-in-context).

\*   Structured Outputs API shape is different. Instead of `response\_format`, use `text.format` in Responses. Learn more in the \[Structured Outputs](/docs/guides/structured-outputs) guide.

\*   The function-calling API shape is different, both for the function config on the request, and function calls sent back in the response. See the full difference in the \[function calling guide](/docs/guides/function-calling).

\*   The Responses SDK has an `output\_text` helper, which the Chat Completions SDK does not have.

\*   In Chat Completions, conversation state must be managed manually. The Responses API has compatibility with the \[Conversations API](/docs/guides/migrate-to-responses) for persistent conversations, or the ability to pass a `previous\_response\_id` to easily chain Responses together.



Migrating from Chat Completions

-------------------------------



\### 1\\. Update generation endpoints



Start by updating your generation endpoints from `post /v1/chat/completions` to `post /v1/responses`.



If you are not using functions or multimodal inputs, then you're done! Simple message inputs are compatible from one API to the other:



Web search tool



```

INPUT='\[

&nbsp; { "role": "system", "content": "You are a helpful assistant." },

&nbsp; { "role": "user", "content": "Hello!" }

]'



curl -s https://api.openai.com/v1/chat/completions \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -H "Authorization: Bearer $OPENAI\_API\_KEY" \\

&nbsp; -d "{

&nbsp;   \\"model\\": \\"gpt-5\\",

&nbsp;   \\"messages\\": $INPUT

&nbsp; }"



curl -s https://api.openai.com/v1/responses \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -H "Authorization: Bearer $OPENAI\_API\_KEY" \\

&nbsp; -d "{

&nbsp;   \\"model\\": \\"gpt-5\\",

&nbsp;   \\"input\\": $INPUT

&nbsp; }"

```



```

const context = \[

&nbsp; { role: 'system', content: 'You are a helpful assistant.' },

&nbsp; { role: 'user', content: 'Hello!' }

];



const completion = await client.chat.completions.create({

&nbsp; model: 'gpt-5',

&nbsp; messages: messages

});



const response = await client.responses.create({

&nbsp; model: "gpt-5",

&nbsp; input: context

});

```



```

context = \[

&nbsp; { "role": "system", "content": "You are a helpful assistant." },

&nbsp; { "role": "user", "content": "Hello!" }

]



completion = client.chat.completions.create(

&nbsp; model="gpt-5",

&nbsp; messages=messages

)



response = client.responses.create(

&nbsp; model="gpt-5",

&nbsp; input=context

)

```



Chat Completions



With Chat Completions, you need to create an array of messages that specify different roles and content for each role.



Generate text from a model



```

import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI\_API\_KEY });



const completion = await client.chat.completions.create({

&nbsp; model: 'gpt-5',

&nbsp; messages: \[

&nbsp;   { 'role': 'system', 'content': 'You are a helpful assistant.' },

&nbsp;   { 'role': 'user', 'content': 'Hello!' }

&nbsp; ]

});

console.log(completion.choices\[0].message.content);

```



```

from openai import OpenAI

client = OpenAI()



completion = client.chat.completions.create(

&nbsp;   model="gpt-5",

&nbsp;   messages=\[

&nbsp;       {"role": "system", "content": "You are a helpful assistant."},

&nbsp;       {"role": "user", "content": "Hello!"}

&nbsp;   ]

)

print(completion.choices\[0].message.content)

```



```

curl https://api.openai.com/v1/chat/completions \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -H "Authorization: Bearer $OPENAI\_API\_KEY" \\

&nbsp; -d '{

&nbsp;     "model": "gpt-5",

&nbsp;     "messages": \[

&nbsp;         {"role": "system", "content": "You are a helpful assistant."},

&nbsp;         {"role": "user", "content": "Hello!"}

&nbsp;     ]

&nbsp; }'

```



Responses



With Responses, you can separate instructions and input at the top-level. The API shape is similar to Chat Completions but has cleaner semantics.



Generate text from a model



```

import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI\_API\_KEY });



const response = await client.responses.create({

&nbsp; model: 'gpt-5',

&nbsp; instructions: 'You are a helpful assistant.',

&nbsp; input: 'Hello!'

});



console.log(response.output\_text);

```



```

from openai import OpenAI

client = OpenAI()



response = client.responses.create(

&nbsp;   model="gpt-5",

&nbsp;   instructions="You are a helpful assistant.",

&nbsp;   input="Hello!"

)

print(response.output\_text)

```



```

curl https://api.openai.com/v1/responses \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -H "Authorization: Bearer $OPENAI\_API\_KEY" \\

&nbsp; -d '{

&nbsp;     "model": "gpt-5",

&nbsp;     "instructions": "You are a helpful assistant.",

&nbsp;     "input": "Hello!"

&nbsp; }'

```



\### 2\\. Update item definitions



Chat Completions



With Chat Completions, you need to create an array of messages that specify different roles and content for each role.



Generate text from a model



```

import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI\_API\_KEY });



const completion = await client.chat.completions.create({

&nbsp; model: 'gpt-5',

&nbsp; messages: \[

&nbsp;   { 'role': 'system', 'content': 'You are a helpful assistant.' },

&nbsp;   { 'role': 'user', 'content': 'Hello!' }

&nbsp; ]

});

console.log(completion.choices\[0].message.content);

```



```

from openai import OpenAI

client = OpenAI()



completion = client.chat.completions.create(

&nbsp;   model="gpt-5",

&nbsp;   messages=\[

&nbsp;       {"role": "system", "content": "You are a helpful assistant."},

&nbsp;       {"role": "user", "content": "Hello!"}

&nbsp;   ]

)

print(completion.choices\[0].message.content)

```



```

curl https://api.openai.com/v1/chat/completions \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -H "Authorization: Bearer $OPENAI\_API\_KEY" \\

&nbsp; -d '{

&nbsp;     "model": "gpt-5",

&nbsp;     "messages": \[

&nbsp;         {"role": "system", "content": "You are a helpful assistant."},

&nbsp;         {"role": "user", "content": "Hello!"}

&nbsp;     ]

&nbsp; }'

```



Responses



With Responses, you can separate instructions and input at the top-level. The API shape is similar to Chat Completions but has cleaner semantics.



Generate text from a model



```

import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI\_API\_KEY });



const response = await client.responses.create({

&nbsp; model: 'gpt-5',

&nbsp; instructions: 'You are a helpful assistant.',

&nbsp; input: 'Hello!'

});



console.log(response.output\_text);

```



```

from openai import OpenAI

client = OpenAI()



response = client.responses.create(

&nbsp;   model="gpt-5",

&nbsp;   instructions="You are a helpful assistant.",

&nbsp;   input="Hello!"

)

print(response.output\_text)

```



```

curl https://api.openai.com/v1/responses \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -H "Authorization: Bearer $OPENAI\_API\_KEY" \\

&nbsp; -d '{

&nbsp;     "model": "gpt-5",

&nbsp;     "instructions": "You are a helpful assistant.",

&nbsp;     "input": "Hello!"

&nbsp; }'

```



\### 3\\. Update multi-turn conversations



If you have multi-turn conversations in your application, update your context logic.



Chat Completions



In Chat Completions, you have to store and manage context yourself.



Multi-turn conversation



```

let messages = \[

&nbsp;   { 'role': 'system', 'content': 'You are a helpful assistant.' },

&nbsp;   { 'role': 'user', 'content': 'What is the capital of France?' }

&nbsp; ];

const res1 = await client.chat.completions.create({

&nbsp; model: 'gpt-5',

&nbsp; messages

});



messages = messages.concat(\[res1.choices\[0].message]);

messages.push({ 'role': 'user', 'content': 'And its population?' });



const res2 = await client.chat.completions.create({

&nbsp; model: 'gpt-5',

&nbsp; messages

});

```



```

messages = \[

&nbsp;   {"role": "system", "content": "You are a helpful assistant."},

&nbsp;   {"role": "user", "content": "What is the capital of France?"}

]

res1 = client.chat.completions.create(model="gpt-5", messages=messages)



messages += \[res1.choices\[0].message]

messages += \[{"role": "user", "content": "And its population?"}]



res2 = client.chat.completions.create(model="gpt-5", messages=messages)

```



Responses



With responses, the pattern is similar, you can pass outputs from one response to the input of another.



Multi-turn conversation



```

context = \[

&nbsp;   { "role": "role", "content": "What is the capital of France?" }

]

res1 = client.responses.create(

&nbsp;   model="gpt-5",

&nbsp;   input=context,

)



// Append the first response’s output to context

context += res1.output



// Add the next user message

context += \[

&nbsp;   { "role": "role", "content": "And it's population?" }

]



res2 = client.responses.create(

&nbsp;   model="gpt-5",

&nbsp;   input=context,

)

```



```

let context = \[

&nbsp; { role: "role", content: "What is the capital of France?" }

];



const res1 = await client.responses.create({

&nbsp; model: "gpt-5",

&nbsp; input: context,

});



// Append the first response’s output to context

context = context.concat(res1.output);



// Add the next user message

context.push({ role: "role", content: "And its population?" });



const res2 = await client.responses.create({

&nbsp; model: "gpt-5",

&nbsp; input: context,

});

```



As a simplification, we've also built a way to simply reference inputs and outputs from a previous response by passing its id. You can use \\`previous\\\_response\\\_id\\` to form chains of responses that build upon one other or create forks in a history.



Multi-turn conversation



```

const res1 = await client.responses.create({

&nbsp; model: 'gpt-5',

&nbsp; input: 'What is the capital of France?',

&nbsp; store: true

});



const res2 = await client.responses.create({

&nbsp; model: 'gpt-5',

&nbsp; input: 'And its population?',

&nbsp; previous\_response\_id: res1.id,

&nbsp; store: true

});

```



```

res1 = client.responses.create(

&nbsp;   model="gpt-5",

&nbsp;   input="What is the capital of France?",

&nbsp;   store=True

)



res2 = client.responses.create(

&nbsp;   model="gpt-5",

&nbsp;   input="And its population?",

&nbsp;   previous\_response\_id=res1.id,

&nbsp;   store=True

)

```



\### 4\\. Decide when to use statefulness



Some organizations—such as those with Zero Data Retention (ZDR) requirements—cannot use the Responses API in a stateful way due to compliance or data retention policies. To support these cases, OpenAI offers encrypted reasoning items, allowing you to keep your workflow stateless while still benefiting from reasoning items.



To disable statefulness, but still take advantage of reasoning:



\*   set `store: false` in the \[store field](/docs/api-reference/responses/create#responses\_create-store)

\*   add `\["reasoning.encrypted\_content"]` to the \[include field](/docs/api-reference/responses/create#responses\_create-include)



The API will then return an encrypted version of the reasoning tokens, which you can pass back in future requests just like regular reasoning items. For ZDR organizations, OpenAI enforces store=false automatically. When a request includes encrypted\\\_content, it is decrypted in-memory (never written to disk), used for generating the next response, and then securely discarded. Any new reasoning tokens are immediately encrypted and returned to you, ensuring no intermediate state is ever persisted.



\### 5\\. Update function definitions



There are two minor, but notable, differences in how functions are defined between Chat Completions and Responses.



1\.  In Chat Completions, functions are defined using externally tagged polymorphism, whereas in Responses, they are internally-tagged.

2\.  In Chat Completions, functions are non-strict by default, whereas in the Responses API, functions \_are\_ strict by default.



The Responses API function example on the right is functionally equivalent to the Chat Completions example on the left.



Chat Completions API



```

{

&nbsp; "type": "function",

&nbsp; "function": {

&nbsp;   "name": "get\_weather",

&nbsp;   "description": "Determine weather in my location",

&nbsp;   "strict": true,

&nbsp;   "parameters": {

&nbsp;     "type": "object",

&nbsp;     "properties": {

&nbsp;       "location": {

&nbsp;         "type": "string",

&nbsp;       },

&nbsp;     },

&nbsp;     "additionalProperties": false,

&nbsp;     "required": \[

&nbsp;       "location",

&nbsp;       "unit"

&nbsp;     ]

&nbsp;   }

&nbsp; }

}

```



Responses API



```

{

&nbsp; "type": "function",

&nbsp; "name": "get\_weather",

&nbsp; "description": "Determine weather in my location",

&nbsp; "parameters": {

&nbsp;   "type": "object",

&nbsp;   "properties": {

&nbsp;     "location": {

&nbsp;       "type": "string",

&nbsp;     },

&nbsp;   },

&nbsp;   "additionalProperties": false,

&nbsp;   "required": \[

&nbsp;     "location",

&nbsp;     "unit"

&nbsp;   ]

&nbsp; }

}

```



\#### Follow function-calling best practices



In Responses, tool calls and their outputs are two distinct types of Items that are correlated using a `call\_id`. See the \[tool calling docs](/docs/guides/function-calling#function-tool-example) for more detail on how function calling works in Responses.



\### 6\\. Update Structured Outputs definition



In the Responses API, defining structured outputs have moved from `response\_format` to `text.format`:



Chat Completions



Structured Outputs



```

curl https://api.openai.com/v1/chat/completions \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -H "Authorization: Bearer $OPENAI\_API\_KEY" \\

&nbsp; -d '{

&nbsp; "model": "gpt-5",

&nbsp; "messages": \[

&nbsp;   {

&nbsp;     "role": "user",

&nbsp;     "content": "Jane, 54 years old",

&nbsp;   }

&nbsp; ],

&nbsp; "response\_format": {

&nbsp;   "type": "json\_schema",

&nbsp;   "json\_schema": {

&nbsp;     "name": "person",

&nbsp;     "strict": true,

&nbsp;     "schema": {

&nbsp;       "type": "object",

&nbsp;       "properties": {

&nbsp;         "name": {

&nbsp;           "type": "string",

&nbsp;           "minLength": 1

&nbsp;         },

&nbsp;         "age": {

&nbsp;           "type": "number",

&nbsp;           "minimum": 0,

&nbsp;           "maximum": 130

&nbsp;         }

&nbsp;       },

&nbsp;       "required": \[

&nbsp;         "name",

&nbsp;         "age"

&nbsp;       ],

&nbsp;       "additionalProperties": false

&nbsp;     }

&nbsp;   }

&nbsp; },

&nbsp; "verbosity": "medium",

&nbsp; "reasoning\_effort": "medium"

}'

```



```

from openai import OpenAI

client = OpenAI()



response = client.chat.completions.create(

&nbsp; model="gpt-5",

&nbsp; messages=\[

&nbsp;   {

&nbsp;     "role": "user",

&nbsp;     "content": "Jane, 54 years old",

&nbsp;   }

&nbsp; ],

&nbsp; response\_format={

&nbsp;   "type": "json\_schema",

&nbsp;   "json\_schema": {

&nbsp;     "name": "person",

&nbsp;     "strict": True,

&nbsp;     "schema": {

&nbsp;       "type": "object",

&nbsp;       "properties": {

&nbsp;         "name": {

&nbsp;           "type": "string",

&nbsp;           "minLength": 1

&nbsp;         },

&nbsp;         "age": {

&nbsp;           "type": "number",

&nbsp;           "minimum": 0,

&nbsp;           "maximum": 130

&nbsp;         }

&nbsp;       },

&nbsp;       "required": \[

&nbsp;         "name",

&nbsp;         "age"

&nbsp;       ],

&nbsp;       "additionalProperties": False

&nbsp;     }

&nbsp;   }

&nbsp; },

&nbsp; verbosity="medium",

&nbsp; reasoning\_effort="medium"

)

```



```

const completion = await openai.chat.completions.create({

&nbsp; model: "gpt-5",

&nbsp; messages: \[

&nbsp;   {

&nbsp;     "role": "user",

&nbsp;     "content": "Jane, 54 years old",

&nbsp;   }

&nbsp; ],

&nbsp; response\_format: {

&nbsp;   type: "json\_schema",

&nbsp;   json\_schema: {

&nbsp;     name: "person",

&nbsp;     strict: true,

&nbsp;     schema: {

&nbsp;       type: "object",

&nbsp;       properties: {

&nbsp;         name: {

&nbsp;           type: "string",

&nbsp;           minLength: 1

&nbsp;         },

&nbsp;         age: {

&nbsp;           type: "number",

&nbsp;           minimum: 0,

&nbsp;           maximum: 130

&nbsp;         }

&nbsp;       },

&nbsp;       required: \[

&nbsp;         name,

&nbsp;         age

&nbsp;       ],

&nbsp;       additionalProperties: false

&nbsp;     }

&nbsp;   }

&nbsp; },

&nbsp; verbosity: "medium",

&nbsp; reasoning\_effort: "medium"

});

```



Responses



Structured Outputs



```

curl https://api.openai.com/v1/responses \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -H "Authorization: Bearer $OPENAI\_API\_KEY" \\

&nbsp; -d '{

&nbsp; "model": "gpt-5",

&nbsp; "input": "Jane, 54 years old",

&nbsp; "text": {

&nbsp;   "format": {

&nbsp;     "type": "json\_schema",

&nbsp;     "name": "person",

&nbsp;     "strict": true,

&nbsp;     "schema": {

&nbsp;       "type": "object",

&nbsp;       "properties": {

&nbsp;         "name": {

&nbsp;           "type": "string",

&nbsp;           "minLength": 1

&nbsp;         },

&nbsp;         "age": {

&nbsp;           "type": "number",

&nbsp;           "minimum": 0,

&nbsp;           "maximum": 130

&nbsp;         }

&nbsp;       },

&nbsp;       "required": \[

&nbsp;         "name",

&nbsp;         "age"

&nbsp;       ],

&nbsp;       "additionalProperties": false

&nbsp;     }

&nbsp;   }

&nbsp; }

}'

```



```

response = client.responses.create(

&nbsp; model="gpt-5",

&nbsp; input="Jane, 54 years old", 

&nbsp; text={

&nbsp;   "format": {

&nbsp;     "type": "json\_schema",

&nbsp;     "name": "person",

&nbsp;     "strict": True,

&nbsp;     "schema": {

&nbsp;       "type": "object",

&nbsp;       "properties": {

&nbsp;         "name": {

&nbsp;           "type": "string",

&nbsp;           "minLength": 1

&nbsp;         },

&nbsp;         "age": {

&nbsp;           "type": "number",

&nbsp;           "minimum": 0,

&nbsp;           "maximum": 130

&nbsp;         }

&nbsp;       },

&nbsp;       "required": \[

&nbsp;         "name",

&nbsp;         "age"

&nbsp;       ],

&nbsp;       "additionalProperties": False

&nbsp;     }

&nbsp;   }

&nbsp; }

)

```



```

const response = await openai.responses.create({

&nbsp; model: "gpt-5",

&nbsp; input: "Jane, 54 years old",

&nbsp; text: {

&nbsp;   format: {

&nbsp;     type: "json\_schema",

&nbsp;     name: "person",

&nbsp;     strict: true,

&nbsp;     schema: {

&nbsp;       type: "object",

&nbsp;       properties: {

&nbsp;         name: {

&nbsp;           type: "string",

&nbsp;           minLength: 1

&nbsp;         },

&nbsp;         age: {

&nbsp;           type: "number",

&nbsp;           minimum: 0,

&nbsp;           maximum: 130

&nbsp;         }

&nbsp;       },

&nbsp;       required: \[

&nbsp;         name,

&nbsp;         age

&nbsp;       ],

&nbsp;       additionalProperties: false

&nbsp;     }

&nbsp;   },

&nbsp; }

});

```



\### 7\\. Upgrade to native tools



If your application has use cases that would benefit from OpenAI's native \[tools](/docs/guides/tools), you can update your tool calls to use OpenAI's tools out of the box.



Chat Completions



With Chat Completions, you cannot use OpenAI's tools natively and have to write your own.



Web search tool



```

async function web\_search(query) {

&nbsp;   const fetch = (await import('node-fetch')).default;

&nbsp;   const res = await fetch(`https://api.example.com/search?q=${query}`);

&nbsp;   const data = await res.json();

&nbsp;   return data.results;

}



const completion = await client.chat.completions.create({

&nbsp; model: 'gpt-5',

&nbsp; messages: \[

&nbsp;   { role: 'system', content: 'You are a helpful assistant.' },

&nbsp;   { role: 'user', content: 'Who is the current president of France?' }

&nbsp; ],

&nbsp; functions: \[

&nbsp;   {

&nbsp;     name: 'web\_search',

&nbsp;     description: 'Search the web for information',

&nbsp;     parameters: {

&nbsp;       type: 'object',

&nbsp;       properties: { query: { type: 'string' } },

&nbsp;       required: \['query']

&nbsp;     }

&nbsp;   }

&nbsp; ]

});

```



```

import requests



def web\_search(query):

&nbsp;   r = requests.get(f"https://api.example.com/search?q={query}")

&nbsp;   return r.json().get("results", \[])



completion = client.chat.completions.create(

&nbsp;   model="gpt-5",

&nbsp;   messages=\[

&nbsp;       {"role": "system", "content": "You are a helpful assistant."},

&nbsp;       {"role": "user", "content": "Who is the current president of France?"}

&nbsp;   ],

&nbsp;   functions=\[

&nbsp;       {

&nbsp;           "name": "web\_search",

&nbsp;           "description": "Search the web for information",

&nbsp;           "parameters": {

&nbsp;               "type": "object",

&nbsp;               "properties": {"query": {"type": "string"}},

&nbsp;               "required": \["query"]

&nbsp;           }

&nbsp;       }

&nbsp;   ]

)

```



```

curl https://api.example.com/search \\

&nbsp; -G \\

&nbsp; --data-urlencode "q=your+search+term" \\

&nbsp; --data-urlencode "key=$SEARCH\_API\_KEY"

```



Responses



With Responses, you can simply specify the tools that you are interested in.



Web search tool



```

const answer = await client.responses.create({

&nbsp;   model: 'gpt-5',

&nbsp;   input: 'Who is the current president of France?',

&nbsp;   tools: \[{ type: 'web\_search' }]

});



console.log(answer.output\_text);

```



```

answer = client.responses.create(

&nbsp;   model="gpt-5",

&nbsp;   input="Who is the current president of France?",

&nbsp;   tools=\[{"type": "web\_search\_preview"}]

)



print(answer.output\_text)

```



```

curl https://api.openai.com/v1/responses \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -H "Authorization: Bearer $OPENAI\_API\_KEY" \\

&nbsp; -d '{

&nbsp;   "model": "gpt-5",

&nbsp;   "input": "Who is the current president of France?",

&nbsp;   "tools": \[{"type": "web\_search"}]

&nbsp; }'

```



Incremental migration

---------------------



The Responses API is a superset of the Chat Completions API. The Chat Completions API will also continue to be supported. As such, you can incrementally adopt the Responses API if desired. You can migrate user flows who would benefit from improved reasoning models to the Responses API while keeping other flows on the Chat Completions API until you're ready for a full migration.



As a best practice, we encourage all users to migrate to the Responses API to take advantage of the latest features and improvements from OpenAI.



Assistants API

--------------



Based on developer feedback from the \[Assistants API](/docs/api-reference/assistants) beta, we've incorporated key improvements into the Responses API to make it more flexible, faster, and easier to use. The Responses API represents the future direction for building agents on OpenAI.



We now have Assistant-like and Thread-like objects in the Responses API. Learn more in the \[migration guide](/docs/guides/assistants/migration). As of August 26th, 2025, we're deprecating the Assistants API, with a sunset date of August 26, 2026.



Was this page useful?

