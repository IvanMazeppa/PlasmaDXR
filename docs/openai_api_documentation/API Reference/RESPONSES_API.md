/

Dashboard

Docs

API reference

Responses

OpenAI's most advanced interface for generating model responses. Supports text and image inputs, and text outputs. Create stateful interactions with the model, using the output of previous responses as input. Extend the model's capabilities with built-in tools for file search, web search, computer use, and more. Allow the model access to external systems and data using function calling.



Related guides:



Quickstart

Text inputs and outputs

Image inputs

Structured Outputs

Function calling

Conversation state

Extend the models with tools

Create a model response

post

&nbsp;

https://api.openai.com/v1/responses

Creates a model response. Provide text or image inputs to generate text or JSON outputs. Have the model call your own custom code or use built-in tools like web search or file search to use your own data as input for the model's response.



Request body

background

boolean



Optional

Defaults to false

Whether to run the model response in the background. Learn more.



conversation

string or object



Optional

Defaults to null

The conversation that this response belongs to. Items from this conversation are prepended to input\_items for this response request. Input items and output items from this response are automatically added to this conversation after this response completes.





Show possible types

include

array



Optional

Specify additional output data to include in the model response. Currently supported values are:



web\_search\_call.action.sources: Include the sources of the web search tool call.

code\_interpreter\_call.outputs: Includes the outputs of python code execution in code interpreter tool call items.

computer\_call\_output.output.image\_url: Include image urls from the computer call output.

file\_search\_call.results: Include the search results of the file search tool call.

message.input\_image.image\_url: Include image urls from the input message.

message.output\_text.logprobs: Include logprobs with assistant messages.

reasoning.encrypted\_content: Includes an encrypted version of reasoning tokens in reasoning item outputs. This enables reasoning items to be used in multi-turn conversations when using the Responses API statelessly (like when the store parameter is set to false, or when an organization is enrolled in the zero data retention program).

input

string or array



Optional

Text, image, or file inputs to the model, used to generate a response.



Learn more:



Text inputs and outputs

Image inputs

File inputs

Conversation state

Function calling



Show possible types

instructions

string



Optional

A system (or developer) message inserted into the model's context.



When using along with previous\_response\_id, the instructions from a previous response will not be carried over to the next response. This makes it simple to swap out system (or developer) messages in new responses.



max\_output\_tokens

integer



Optional

An upper bound for the number of tokens that can be generated for a response, including visible output tokens and reasoning tokens.



max\_tool\_calls

integer



Optional

The maximum number of total calls to built-in tools that can be processed in a response. This maximum number applies across all built-in tool calls, not per individual tool. Any further attempts to call a tool by the model will be ignored.



metadata

map



Optional

Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard.



Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.



model

string



Optional

Model ID used to generate the response, like gpt-4o or o3. OpenAI offers a wide range of models with different capabilities, performance characteristics, and price points. Refer to the model guide to browse and compare available models.



parallel\_tool\_calls

boolean



Optional

Defaults to true

Whether to allow the model to run tool calls in parallel.



previous\_response\_id

string



Optional

The unique ID of the previous response to the model. Use this to create multi-turn conversations. Learn more about conversation state. Cannot be used in conjunction with conversation.



prompt

object



Optional

Reference to a prompt template and its variables. Learn more.





Show properties

prompt\_cache\_key

string



Optional

Used by OpenAI to cache responses for similar requests to optimize your cache hit rates. Replaces the user field. Learn more.



prompt\_cache\_retention

string



Optional

The retention policy for the prompt cache. Set to 24h to enable extended prompt caching, which keeps cached prefixes active for longer, up to a maximum of 24 hours. Learn more.



reasoning

object



Optional

gpt-5 and o-series models only



Configuration options for reasoning models.





Show properties

safety\_identifier

string



Optional

A stable identifier used to help detect users of your application that may be violating OpenAI's usage policies. The IDs should be a string that uniquely identifies each user. We recommend hashing their username or email address, in order to avoid sending us any identifying information. Learn more.



service\_tier

string



Optional

Defaults to auto

Specifies the processing type used for serving the request.



If set to 'auto', then the request will be processed with the service tier configured in the Project settings. Unless otherwise configured, the Project will use 'default'.

If set to 'default', then the request will be processed with the standard pricing and performance for the selected model.

If set to 'flex' or 'priority', then the request will be processed with the corresponding service tier.

When not set, the default behavior is 'auto'.

When the service\_tier parameter is set, the response body will include the service\_tier value based on the processing mode actually used to serve the request. This response value may be different from the value set in the parameter.



store

boolean



Optional

Defaults to true

Whether to store the generated model response for later retrieval via API.



stream

boolean



Optional

Defaults to false

If set to true, the model response data will be streamed to the client as it is generated using server-sent events. See the Streaming section below for more information.



stream\_options

object



Optional

Defaults to null

Options for streaming responses. Only set this when you set stream: true.





Show properties

temperature

number



Optional

Defaults to 1

What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top\_p but not both.



text

object



Optional

Configuration options for a text response from the model. Can be plain text or structured JSON data. Learn more:



Text inputs and outputs

Structured Outputs



Show properties

tool\_choice

string or object



Optional

How the model should select which tool (or tools) to use when generating a response. See the tools parameter to see how to specify which tools the model can call.





Show possible types

tools

array



Optional

An array of tools the model may call while generating a response. You can specify which tool to use by setting the tool\_choice parameter.



We support the following categories of tools:



Built-in tools: Tools that are provided by OpenAI that extend the model's capabilities, like web search or file search. Learn more about built-in tools.

MCP Tools: Integrations with third-party systems via custom MCP servers or predefined connectors such as Google Drive and SharePoint. Learn more about MCP Tools.

Function calls (custom tools): Functions that are defined by you, enabling the model to call your own code with strongly typed arguments and outputs. Learn more about function calling. You can also use custom tools to call your own code.



Show possible types

top\_logprobs

integer



Optional

An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability.



top\_p

number



Optional

Defaults to 1

An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top\_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.



We generally recommend altering this or temperature but not both.



truncation

string



Optional

Defaults to disabled

The truncation strategy to use for the model response.



auto: If the input to this Response exceeds the model's context window size, the model will truncate the response to fit the context window by dropping items from the beginning of the conversation.

disabled (default): If the input size will exceed the context window size for a model, the request will fail with a 400 error.

user

Deprecated

string



Optional

This field is being replaced by safety\_identifier and prompt\_cache\_key. Use prompt\_cache\_key instead to maintain caching optimizations. A stable identifier for your end-users. Used to boost cache hit rates by better bucketing similar requests and to help OpenAI detect and prevent abuse. Learn more.



Returns

Returns a Response object.



Text input

Image input

File input

Web search

File search

Streaming

Functions

Reasoning

Example request

from openai import OpenAI



client = OpenAI()



response = client.responses.create(

&nbsp; model="gpt-4.1",

&nbsp; input="Tell me a three sentence bedtime story about a unicorn."

)



print(response)

Response

{

&nbsp; "id": "resp\_67ccd2bed1ec8190b14f964abc0542670bb6a6b452d3795b",

&nbsp; "object": "response",

&nbsp; "created\_at": 1741476542,

&nbsp; "status": "completed",

&nbsp; "error": null,

&nbsp; "incomplete\_details": null,

&nbsp; "instructions": null,

&nbsp; "max\_output\_tokens": null,

&nbsp; "model": "gpt-4.1-2025-04-14",

&nbsp; "output": \[

&nbsp;   {

&nbsp;     "type": "message",

&nbsp;     "id": "msg\_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",

&nbsp;     "status": "completed",

&nbsp;     "role": "assistant",

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "type": "output\_text",

&nbsp;         "text": "In a peaceful grove beneath a silver moon, a unicorn named Lumina discovered a hidden pool that reflected the stars. As she dipped her horn into the water, the pool began to shimmer, revealing a pathway to a magical realm of endless night skies. Filled with wonder, Lumina whispered a wish for all who dream to find their own hidden magic, and as she glanced back, her hoofprints sparkled like stardust.",

&nbsp;         "annotations": \[]

&nbsp;       }

&nbsp;     ]

&nbsp;   }

&nbsp; ],

&nbsp; "parallel\_tool\_calls": true,

&nbsp; "previous\_response\_id": null,

&nbsp; "reasoning": {

&nbsp;   "effort": null,

&nbsp;   "summary": null

&nbsp; },

&nbsp; "store": true,

&nbsp; "temperature": 1.0,

&nbsp; "text": {

&nbsp;   "format": {

&nbsp;     "type": "text"

&nbsp;   }

&nbsp; },

&nbsp; "tool\_choice": "auto",

&nbsp; "tools": \[],

&nbsp; "top\_p": 1.0,

&nbsp; "truncation": "disabled",

&nbsp; "usage": {

&nbsp;   "input\_tokens": 36,

&nbsp;   "input\_tokens\_details": {

&nbsp;     "cached\_tokens": 0

&nbsp;   },

&nbsp;   "output\_tokens": 87,

&nbsp;   "output\_tokens\_details": {

&nbsp;     "reasoning\_tokens": 0

&nbsp;   },

&nbsp;   "total\_tokens": 123

&nbsp; },

&nbsp; "user": null,

&nbsp; "metadata": {}

}

Get a model response

get

&nbsp;

https://api.openai.com/v1/responses/{response\_id}

Retrieves a model response with the given ID.



Path parameters

response\_id

string



Required

The ID of the response to retrieve.



Query parameters

include

array



Optional

Additional fields to include in the response. See the include parameter for Response creation above for more information.



include\_obfuscation

boolean



Optional

When true, stream obfuscation will be enabled. Stream obfuscation adds random characters to an obfuscation field on streaming delta events to normalize payload sizes as a mitigation to certain side-channel attacks. These obfuscation fields are included by default, but add a small amount of overhead to the data stream. You can set include\_obfuscation to false to optimize for bandwidth if you trust the network links between your application and the OpenAI API.



starting\_after

integer



Optional

The sequence number of the event after which to start streaming.



stream

boolean



Optional

If set to true, the model response data will be streamed to the client as it is generated using server-sent events. See the Streaming section below for more information.



Returns

The Response object matching the specified ID.



Example request

from openai import OpenAI

client = OpenAI()



response = client.responses.retrieve("resp\_123")

print(response)

Response

{

&nbsp; "id": "resp\_67cb71b351908190a308f3859487620d06981a8637e6bc44",

&nbsp; "object": "response",

&nbsp; "created\_at": 1741386163,

&nbsp; "status": "completed",

&nbsp; "error": null,

&nbsp; "incomplete\_details": null,

&nbsp; "instructions": null,

&nbsp; "max\_output\_tokens": null,

&nbsp; "model": "gpt-4o-2024-08-06",

&nbsp; "output": \[

&nbsp;   {

&nbsp;     "type": "message",

&nbsp;     "id": "msg\_67cb71b3c2b0819084d481baaaf148f206981a8637e6bc44",

&nbsp;     "status": "completed",

&nbsp;     "role": "assistant",

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "type": "output\_text",

&nbsp;         "text": "Silent circuits hum,  \\nThoughts emerge in data streams—  \\nDigital dawn breaks.",

&nbsp;         "annotations": \[]

&nbsp;       }

&nbsp;     ]

&nbsp;   }

&nbsp; ],

&nbsp; "parallel\_tool\_calls": true,

&nbsp; "previous\_response\_id": null,

&nbsp; "reasoning": {

&nbsp;   "effort": null,

&nbsp;   "summary": null

&nbsp; },

&nbsp; "store": true,

&nbsp; "temperature": 1.0,

&nbsp; "text": {

&nbsp;   "format": {

&nbsp;     "type": "text"

&nbsp;   }

&nbsp; },

&nbsp; "tool\_choice": "auto",

&nbsp; "tools": \[],

&nbsp; "top\_p": 1.0,

&nbsp; "truncation": "disabled",

&nbsp; "usage": {

&nbsp;   "input\_tokens": 32,

&nbsp;   "input\_tokens\_details": {

&nbsp;     "cached\_tokens": 0

&nbsp;   },

&nbsp;   "output\_tokens": 18,

&nbsp;   "output\_tokens\_details": {

&nbsp;     "reasoning\_tokens": 0

&nbsp;   },

&nbsp;   "total\_tokens": 50

&nbsp; },

&nbsp; "user": null,

&nbsp; "metadata": {}

}

Delete a model response

delete

&nbsp;

https://api.openai.com/v1/responses/{response\_id}

Deletes a model response with the given ID.



Path parameters

response\_id

string



Required

The ID of the response to delete.



Returns

A success message.



Example request

from openai import OpenAI

client = OpenAI()



response = client.responses.delete("resp\_123")

print(response)

Response

{

&nbsp; "id": "resp\_6786a1bec27481909a17d673315b29f6",

&nbsp; "object": "response",

&nbsp; "deleted": true

}

Cancel a response

post

&nbsp;

https://api.openai.com/v1/responses/{response\_id}/cancel

Cancels a model response with the given ID. Only responses created with the background parameter set to true can be cancelled. Learn more.



Path parameters

response\_id

string



Required

The ID of the response to cancel.



Returns

A Response object.



Example request

from openai import OpenAI

client = OpenAI()



response = client.responses.cancel("resp\_123")

print(response)

Response

{

&nbsp; "id": "resp\_67cb71b351908190a308f3859487620d06981a8637e6bc44",

&nbsp; "object": "response",

&nbsp; "created\_at": 1741386163,

&nbsp; "status": "completed",

&nbsp; "error": null,

&nbsp; "incomplete\_details": null,

&nbsp; "instructions": null,

&nbsp; "max\_output\_tokens": null,

&nbsp; "model": "gpt-4o-2024-08-06",

&nbsp; "output": \[

&nbsp;   {

&nbsp;     "type": "message",

&nbsp;     "id": "msg\_67cb71b3c2b0819084d481baaaf148f206981a8637e6bc44",

&nbsp;     "status": "completed",

&nbsp;     "role": "assistant",

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "type": "output\_text",

&nbsp;         "text": "Silent circuits hum,  \\nThoughts emerge in data streams—  \\nDigital dawn breaks.",

&nbsp;         "annotations": \[]

&nbsp;       }

&nbsp;     ]

&nbsp;   }

&nbsp; ],

&nbsp; "parallel\_tool\_calls": true,

&nbsp; "previous\_response\_id": null,

&nbsp; "reasoning": {

&nbsp;   "effort": null,

&nbsp;   "summary": null

&nbsp; },

&nbsp; "store": true,

&nbsp; "temperature": 1.0,

&nbsp; "text": {

&nbsp;   "format": {

&nbsp;     "type": "text"

&nbsp;   }

&nbsp; },

&nbsp; "tool\_choice": "auto",

&nbsp; "tools": \[],

&nbsp; "top\_p": 1.0,

&nbsp; "truncation": "disabled",

&nbsp; "usage": {

&nbsp;   "input\_tokens": 32,

&nbsp;   "input\_tokens\_details": {

&nbsp;     "cached\_tokens": 0

&nbsp;   },

&nbsp;   "output\_tokens": 18,

&nbsp;   "output\_tokens\_details": {

&nbsp;     "reasoning\_tokens": 0

&nbsp;   },

&nbsp;   "total\_tokens": 50

&nbsp; },

&nbsp; "user": null,

&nbsp; "metadata": {}

}

Compact a response

post

&nbsp;

https://api.openai.com/v1/responses/compact

Runs a compaction pass over a conversation. Compaction returns encrypted, opaque items and the underlying logic may evolve over time.



Request body

model

string



Required

Model ID used to generate the response, like gpt-5 or o3. OpenAI offers a wide range of models with different capabilities, performance characteristics, and price points. Refer to the model guide to browse and compare available models.



input

string or array



Optional

Text, image, or file inputs to the model, used to generate a response





Show possible types

instructions

string



Optional

A system (or developer) message inserted into the model's context. When used along with previous\_response\_id, the instructions from a previous response will not be carried over to the next response. This makes it simple to swap out system (or developer) messages in new responses.



previous\_response\_id

string



Optional

The unique ID of the previous response to the model. Use this to create multi-turn conversations. Learn more about conversation state. Cannot be used in conjunction with conversation.



Returns

A compacted response object.



Learn when and how to compact long-running conversations in the conversation state guide.



Example request

from openai import OpenAI



client = OpenAI()



compacted\_response = client.responses.compact(

&nbsp;   model="gpt-5.1-codex-max",

&nbsp;   input=\[

&nbsp;   {

&nbsp;       "role": "user",

&nbsp;       "content": "Create a simple landing page for a dog petting cafe.",

&nbsp;   },

&nbsp;   # All items returned from previous requests are included here, like reasoning, message, function call, etc.

&nbsp;   {

&nbsp;       "id": "msg\_001",

&nbsp;       "type": "message",

&nbsp;       "status": "completed",

&nbsp;       "content": \[

&nbsp;       {

&nbsp;           "type": "output\_text",

&nbsp;           "annotations": \[],

&nbsp;           "logprobs": \[],

&nbsp;           "text": "Below is a single file, ready-to-use landing page for a dog petting café:...",

&nbsp;       },

&nbsp;       ],

&nbsp;       "role": "assistant",

&nbsp;   },

&nbsp;   ]

)

\# Pass the compacted\_response.output as input to the next request

print(compacted\_response)

Response

{

&nbsp; "id": "resp\_001",

&nbsp; "object": "response.compaction",

&nbsp; "created\_at": 1764967971,

&nbsp; "output": \[

&nbsp;   {

&nbsp;     "id": "msg\_000",

&nbsp;     "type": "message",

&nbsp;     "status": "completed",

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "type": "input\_text",

&nbsp;         "text": "Create a simple landing page for a dog petting cafe."

&nbsp;       }

&nbsp;     ],

&nbsp;     "role": "user"

&nbsp;   },

&nbsp;   {

&nbsp;     "id": "cmp\_001",

&nbsp;     "type": "compaction",

&nbsp;     "encrypted\_content": "gAAAAABpM0Yj-...="

&nbsp;   }

&nbsp; ],

&nbsp; "usage": {

&nbsp;   "input\_tokens": 139,

&nbsp;   "input\_tokens\_details": {

&nbsp;     "cached\_tokens": 0

&nbsp;   },

&nbsp;   "output\_tokens": 438,

&nbsp;   "output\_tokens\_details": {

&nbsp;     "reasoning\_tokens": 64

&nbsp;   },

&nbsp;   "total\_tokens": 577

&nbsp; }

}

List input items

get

&nbsp;

https://api.openai.com/v1/responses/{response\_id}/input\_items

Returns a list of input items for a given response.



Path parameters

response\_id

string



Required

The ID of the response to retrieve input items for.



Query parameters

after

string



Optional

An item ID to list items after, used in pagination.



include

array



Optional

Additional fields to include in the response. See the include parameter for Response creation above for more information.



limit

integer



Optional

Defaults to 20

A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20.



order

string



Optional

The order to return the input items in. Default is desc.



asc: Return the input items in ascending order.

desc: Return the input items in descending order.

Returns

A list of input item objects.



Example request

from openai import OpenAI

client = OpenAI()



response = client.responses.input\_items.list("resp\_123")

print(response.data)

Response

{

&nbsp; "object": "list",

&nbsp; "data": \[

&nbsp;   {

&nbsp;     "id": "msg\_abc123",

&nbsp;     "type": "message",

&nbsp;     "role": "user",

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "type": "input\_text",

&nbsp;         "text": "Tell me a three sentence bedtime story about a unicorn."

&nbsp;       }

&nbsp;     ]

&nbsp;   }

&nbsp; ],

&nbsp; "first\_id": "msg\_abc123",

&nbsp; "last\_id": "msg\_abc123",

&nbsp; "has\_more": false

}

Get input token counts

post

&nbsp;

https://api.openai.com/v1/responses/input\_tokens

Returns input token counts of the request.



Request body

conversation

string or object



Optional

Defaults to null

The conversation that this response belongs to. Items from this conversation are prepended to input\_items for this response request. Input items and output items from this response are automatically added to this conversation after this response completes.





Show possible types

input

string or array



Optional

Text, image, or file inputs to the model, used to generate a response





Show possible types

instructions

string



Optional

A system (or developer) message inserted into the model's context. When used along with previous\_response\_id, the instructions from a previous response will not be carried over to the next response. This makes it simple to swap out system (or developer) messages in new responses.



model

string



Optional

Model ID used to generate the response, like gpt-4o or o3. OpenAI offers a wide range of models with different capabilities, performance characteristics, and price points. Refer to the model guide to browse and compare available models.



parallel\_tool\_calls

boolean



Optional

Whether to allow the model to run tool calls in parallel.



previous\_response\_id

string



Optional

The unique ID of the previous response to the model. Use this to create multi-turn conversations. Learn more about conversation state. Cannot be used in conjunction with conversation.



reasoning

object



Optional

gpt-5 and o-series models only



Configuration options for reasoning models.





Show properties

text

object



Optional

Configuration options for a text response from the model. Can be plain text or structured JSON data. Learn more:



Text inputs and outputs

Structured Outputs



Show properties

tool\_choice

string or object



Optional

How the model should select which tool (or tools) to use when generating a response. See the tools parameter to see how to specify which tools the model can call.





Show possible types

tools

array



Optional

An array of tools the model may call while generating a response. You can specify which tool to use by setting the tool\_choice parameter.





Show possible types

truncation

string



Optional

The truncation strategy to use for the model response. - auto: If the input to this Response exceeds the model's context window size, the model will truncate the response to fit the context window by dropping items from the beginning of the conversation. - disabled (default): If the input size will exceed the context window size for a model, the request will fail with a 400 error.



Returns

The input token counts.



{

&nbsp; object: "response.input\_tokens"

&nbsp; input\_tokens: 123

}

Example request

from openai import OpenAI



client = OpenAI()



response = client.responses.input\_tokens.count(

&nbsp;   model="gpt-5",

&nbsp;   input="Tell me a joke."

)

print(response.input\_tokens)

Response

{

&nbsp; "object": "response.input\_tokens",

&nbsp; "input\_tokens": 11

}

The response object

background

boolean



Whether to run the model response in the background. Learn more.



conversation

object



The conversation that this response belongs to. Input items and output items from this response are automatically added to this conversation.





Show properties

created\_at

number



Unix timestamp (in seconds) of when this Response was created.



error

object



An error object returned when the model fails to generate a Response.





Show properties

id

string



Unique identifier for this Response.



incomplete\_details

object



Details about why the response is incomplete.





Show properties

instructions

string or array



A system (or developer) message inserted into the model's context.



When using along with previous\_response\_id, the instructions from a previous response will not be carried over to the next response. This makes it simple to swap out system (or developer) messages in new responses.





Show possible types

max\_output\_tokens

integer



An upper bound for the number of tokens that can be generated for a response, including visible output tokens and reasoning tokens.



max\_tool\_calls

integer



The maximum number of total calls to built-in tools that can be processed in a response. This maximum number applies across all built-in tool calls, not per individual tool. Any further attempts to call a tool by the model will be ignored.



metadata

map



Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard.



Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.



model

string



Model ID used to generate the response, like gpt-4o or o3. OpenAI offers a wide range of models with different capabilities, performance characteristics, and price points. Refer to the model guide to browse and compare available models.



object

string



The object type of this resource - always set to response.



output

array



An array of content items generated by the model.



The length and order of items in the output array is dependent on the model's response.

Rather than accessing the first item in the output array and assuming it's an assistant message with the content generated by the model, you might consider using the output\_text property where supported in SDKs.



Show possible types

output\_text

string



SDK Only

SDK-only convenience property that contains the aggregated text output from all output\_text items in the output array, if any are present. Supported in the Python and JavaScript SDKs.



parallel\_tool\_calls

boolean



Whether to allow the model to run tool calls in parallel.



previous\_response\_id

string



The unique ID of the previous response to the model. Use this to create multi-turn conversations. Learn more about conversation state. Cannot be used in conjunction with conversation.



prompt

object



Reference to a prompt template and its variables. Learn more.





Show properties

prompt\_cache\_key

string



Used by OpenAI to cache responses for similar requests to optimize your cache hit rates. Replaces the user field. Learn more.



prompt\_cache\_retention

string



The retention policy for the prompt cache. Set to 24h to enable extended prompt caching, which keeps cached prefixes active for longer, up to a maximum of 24 hours. Learn more.



reasoning

object



gpt-5 and o-series models only



Configuration options for reasoning models.





Show properties

safety\_identifier

string



A stable identifier used to help detect users of your application that may be violating OpenAI's usage policies. The IDs should be a string that uniquely identifies each user. We recommend hashing their username or email address, in order to avoid sending us any identifying information. Learn more.



service\_tier

string



Specifies the processing type used for serving the request.



If set to 'auto', then the request will be processed with the service tier configured in the Project settings. Unless otherwise configured, the Project will use 'default'.

If set to 'default', then the request will be processed with the standard pricing and performance for the selected model.

If set to 'flex' or 'priority', then the request will be processed with the corresponding service tier.

When not set, the default behavior is 'auto'.

When the service\_tier parameter is set, the response body will include the service\_tier value based on the processing mode actually used to serve the request. This response value may be different from the value set in the parameter.



status

string



The status of the response generation. One of completed, failed, in\_progress, cancelled, queued, or incomplete.



temperature

number



What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top\_p but not both.



text

object



Configuration options for a text response from the model. Can be plain text or structured JSON data. Learn more:



Text inputs and outputs

Structured Outputs



Show properties

tool\_choice

string or object



How the model should select which tool (or tools) to use when generating a response. See the tools parameter to see how to specify which tools the model can call.





Show possible types

tools

array



An array of tools the model may call while generating a response. You can specify which tool to use by setting the tool\_choice parameter.



We support the following categories of tools:



Built-in tools: Tools that are provided by OpenAI that extend the model's capabilities, like web search or file search. Learn more about built-in tools.

MCP Tools: Integrations with third-party systems via custom MCP servers or predefined connectors such as Google Drive and SharePoint. Learn more about MCP Tools.

Function calls (custom tools): Functions that are defined by you, enabling the model to call your own code with strongly typed arguments and outputs. Learn more about function calling. You can also use custom tools to call your own code.



Show possible types

top\_logprobs

integer



An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability.



top\_p

number



An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top\_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.



We generally recommend altering this or temperature but not both.



truncation

string



The truncation strategy to use for the model response.



auto: If the input to this Response exceeds the model's context window size, the model will truncate the response to fit the context window by dropping items from the beginning of the conversation.

disabled (default): If the input size will exceed the context window size for a model, the request will fail with a 400 error.

usage

object



Represents token usage details including input tokens, output tokens, a breakdown of output tokens, and the total tokens used.





Show properties

user

Deprecated

string



This field is being replaced by safety\_identifier and prompt\_cache\_key. Use prompt\_cache\_key instead to maintain caching optimizations. A stable identifier for your end-users. Used to boost cache hit rates by better bucketing similar requests and to help OpenAI detect and prevent abuse. Learn more.



OBJECT The response object

{

&nbsp; "id": "resp\_67ccd3a9da748190baa7f1570fe91ac604becb25c45c1d41",

&nbsp; "object": "response",

&nbsp; "created\_at": 1741476777,

&nbsp; "status": "completed",

&nbsp; "error": null,

&nbsp; "incomplete\_details": null,

&nbsp; "instructions": null,

&nbsp; "max\_output\_tokens": null,

&nbsp; "model": "gpt-4o-2024-08-06",

&nbsp; "output": \[

&nbsp;   {

&nbsp;     "type": "message",

&nbsp;     "id": "msg\_67ccd3acc8d48190a77525dc6de64b4104becb25c45c1d41",

&nbsp;     "status": "completed",

&nbsp;     "role": "assistant",

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "type": "output\_text",

&nbsp;         "text": "The image depicts a scenic landscape with a wooden boardwalk or pathway leading through lush, green grass under a blue sky with some clouds. The setting suggests a peaceful natural area, possibly a park or nature reserve. There are trees and shrubs in the background.",

&nbsp;         "annotations": \[]

&nbsp;       }

&nbsp;     ]

&nbsp;   }

&nbsp; ],

&nbsp; "parallel\_tool\_calls": true,

&nbsp; "previous\_response\_id": null,

&nbsp; "reasoning": {

&nbsp;   "effort": null,

&nbsp;   "summary": null

&nbsp; },

&nbsp; "store": true,

&nbsp; "temperature": 1,

&nbsp; "text": {

&nbsp;   "format": {

&nbsp;     "type": "text"

&nbsp;   }

&nbsp; },

&nbsp; "tool\_choice": "auto",

&nbsp; "tools": \[],

&nbsp; "top\_p": 1,

&nbsp; "truncation": "disabled",

&nbsp; "usage": {

&nbsp;   "input\_tokens": 328,

&nbsp;   "input\_tokens\_details": {

&nbsp;     "cached\_tokens": 0

&nbsp;   },

&nbsp;   "output\_tokens": 52,

&nbsp;   "output\_tokens\_details": {

&nbsp;     "reasoning\_tokens": 0

&nbsp;   },

&nbsp;   "total\_tokens": 380

&nbsp; },

&nbsp; "user": null,

&nbsp; "metadata": {}

}

The input item list

A list of Response items.



data

array



A list of items used to generate this response.





Show possible types

first\_id

string



The ID of the first item in the list.



has\_more

boolean



Whether there are more items available.



last\_id

string



The ID of the last item in the list.



object

string



The type of object returned, must be list.



OBJECT The input item list

{

&nbsp; "object": "list",

&nbsp; "data": \[

&nbsp;   {

&nbsp;     "id": "msg\_abc123",

&nbsp;     "type": "message",

&nbsp;     "role": "user",

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "type": "input\_text",

&nbsp;         "text": "Tell me a three sentence bedtime story about a unicorn."

&nbsp;       }

&nbsp;     ]

&nbsp;   }

&nbsp; ],

&nbsp; "first\_id": "msg\_abc123",

&nbsp; "last\_id": "msg\_abc123",

&nbsp; "has\_more": false

}

The compacted response object

created\_at

integer



Unix timestamp (in seconds) when the compacted conversation was created.



id

string



The unique identifier for the compacted response.



object

string



The object type. Always response.compaction.



output

array



The compacted list of output items.





Show possible types

usage

object



Represents token usage details including input tokens, output tokens, a breakdown of output tokens, and the total tokens used.





Show properties

OBJECT The compacted response object

{

&nbsp; "id": "resp\_001",

&nbsp; "object": "response.compaction",

&nbsp; "output": \[

&nbsp;   {

&nbsp;     "type": "message",

&nbsp;     "role": "user",

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "type": "input\_text",

&nbsp;         "text": "Summarize our launch checklist from last week."

&nbsp;       }

&nbsp;     ]

&nbsp;   },

&nbsp;   {

&nbsp;     "type": "message",

&nbsp;     "role": "user",

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "type": "input\_text",

&nbsp;         "text": "You are performing a CONTEXT CHECKPOINT COMPACTION..."

&nbsp;       }

&nbsp;     ]

&nbsp;   },

&nbsp;   {

&nbsp;     "type": "compaction",

&nbsp;     "id": "cmp\_001",

&nbsp;     "encrypted\_content": "encrypted-summary"

&nbsp;   }

&nbsp; ],

&nbsp; "created\_at": 1731459200,

&nbsp; "usage": {

&nbsp;   "input\_tokens": 42897,

&nbsp;   "output\_tokens": 12000,

&nbsp;   "total\_tokens": 54912

&nbsp; }

}

Previous

Introduction

Next

Conversations

