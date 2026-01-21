/

Dashboard

Docs

API reference

Evals

Create, manage, and run evals in the OpenAI platform. Related guide: Evals



Create eval

post

&nbsp;

https://api.openai.com/v1/evals

Create the structure of an evaluation that can be used to test a model's performance. An evaluation is a set of testing criteria and the config for a data source, which dictates the schema of the data used in the evaluation. After creating an evaluation, you can run it on different models and model parameters. We support several types of graders and datasources. For more information, see the Evals guide.



Request body

data\_source\_config

object



Required

The configuration for the data source used for the evaluation runs. Dictates the schema of the data used in the evaluation.





Show possible types

testing\_criteria

array



Required

A list of graders for all eval runs in this group. Graders can reference variables in the data source using double curly braces notation, like {{item.variable\_name}}. To reference the model's output, use the sample namespace (ie, {{sample.output\_text}}).





Show possible types

metadata

map



Optional

Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard.



Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.



name

string



Optional

The name of the evaluation.



Returns

The created Eval object.



Example request

import OpenAI from "openai";



const openai = new OpenAI();



const evalObj = await openai.evals.create({

&nbsp; name: "Sentiment",

&nbsp; data\_source\_config: {

&nbsp;   type: "stored\_completions",

&nbsp;   metadata: { usecase: "chatbot" }

&nbsp; },

&nbsp; testing\_criteria: \[

&nbsp;   {

&nbsp;     type: "label\_model",

&nbsp;     model: "o3-mini",

&nbsp;     input: \[

&nbsp;       { role: "developer", content: "Classify the sentiment of the following statement as one of 'positive', 'neutral', or 'negative'" },

&nbsp;       { role: "user", content: "Statement: {{item.input}}" }

&nbsp;     ],

&nbsp;     passing\_labels: \["positive"],

&nbsp;     labels: \["positive", "neutral", "negative"],

&nbsp;     name: "Example label grader"

&nbsp;   }

&nbsp; ]

});

console.log(evalObj);

Response

{

&nbsp; "object": "eval",

&nbsp; "id": "eval\_67b7fa9a81a88190ab4aa417e397ea21",

&nbsp; "data\_source\_config": {

&nbsp;   "type": "stored\_completions",

&nbsp;   "metadata": {

&nbsp;     "usecase": "chatbot"

&nbsp;   },

&nbsp;   "schema": {

&nbsp;     "type": "object",

&nbsp;     "properties": {

&nbsp;       "item": {

&nbsp;         "type": "object"

&nbsp;       },

&nbsp;       "sample": {

&nbsp;         "type": "object"

&nbsp;       }

&nbsp;     },

&nbsp;     "required": \[

&nbsp;       "item",

&nbsp;       "sample"

&nbsp;     ]

&nbsp; },

&nbsp; "testing\_criteria": \[

&nbsp;   {

&nbsp;     "name": "Example label grader",

&nbsp;     "type": "label\_model",

&nbsp;     "model": "o3-mini",

&nbsp;     "input": \[

&nbsp;       {

&nbsp;         "type": "message",

&nbsp;         "role": "developer",

&nbsp;         "content": {

&nbsp;           "type": "input\_text",

&nbsp;           "text": "Classify the sentiment of the following statement as one of positive, neutral, or negative"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "type": "message",

&nbsp;         "role": "user",

&nbsp;         "content": {

&nbsp;           "type": "input\_text",

&nbsp;           "text": "Statement: {{item.input}}"

&nbsp;         }

&nbsp;       }

&nbsp;     ],

&nbsp;     "passing\_labels": \[

&nbsp;       "positive"

&nbsp;     ],

&nbsp;     "labels": \[

&nbsp;       "positive",

&nbsp;       "neutral",

&nbsp;       "negative"

&nbsp;     ]

&nbsp;   }

&nbsp; ],

&nbsp; "name": "Sentiment",

&nbsp; "created\_at": 1740110490,

&nbsp; "metadata": {

&nbsp;   "description": "An eval for sentiment analysis"

&nbsp; }

}

Get an eval

get

&nbsp;

https://api.openai.com/v1/evals/{eval\_id}

Get an evaluation by ID.



Path parameters

eval\_id

string



Required

The ID of the evaluation to retrieve.



Returns

The Eval object matching the specified ID.



Example request

import OpenAI from "openai";



const openai = new OpenAI();



const evalObj = await openai.evals.retrieve("eval\_67abd54d9b0081909a86353f6fb9317a");

console.log(evalObj);

Response

{

&nbsp; "object": "eval",

&nbsp; "id": "eval\_67abd54d9b0081909a86353f6fb9317a",

&nbsp; "data\_source\_config": {

&nbsp;   "type": "custom",

&nbsp;   "schema": {

&nbsp;     "type": "object",

&nbsp;     "properties": {

&nbsp;       "item": {

&nbsp;         "type": "object",

&nbsp;         "properties": {

&nbsp;           "input": {

&nbsp;             "type": "string"

&nbsp;           },

&nbsp;           "ground\_truth": {

&nbsp;             "type": "string"

&nbsp;           }

&nbsp;         },

&nbsp;         "required": \[

&nbsp;           "input",

&nbsp;           "ground\_truth"

&nbsp;         ]

&nbsp;       }

&nbsp;     },

&nbsp;     "required": \[

&nbsp;       "item"

&nbsp;     ]

&nbsp;   }

&nbsp; },

&nbsp; "testing\_criteria": \[

&nbsp;   {

&nbsp;     "name": "String check",

&nbsp;     "id": "String check-2eaf2d8d-d649-4335-8148-9535a7ca73c2",

&nbsp;     "type": "string\_check",

&nbsp;     "input": "{{item.input}}",

&nbsp;     "reference": "{{item.ground\_truth}}",

&nbsp;     "operation": "eq"

&nbsp;   }

&nbsp; ],

&nbsp; "name": "External Data Eval",

&nbsp; "created\_at": 1739314509,

&nbsp; "metadata": {},

}

Update an eval

post

&nbsp;

https://api.openai.com/v1/evals/{eval\_id}

Update certain properties of an evaluation.



Path parameters

eval\_id

string



Required

The ID of the evaluation to update.



Request body

metadata

map



Optional

Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard.



Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.



name

string



Optional

Rename the evaluation.



Returns

The Eval object matching the updated version.



Example request

import OpenAI from "openai";



const openai = new OpenAI();



const updatedEval = await openai.evals.update(

&nbsp; "eval\_67abd54d9b0081909a86353f6fb9317a",

&nbsp; {

&nbsp;   name: "Updated Eval",

&nbsp;   metadata: { description: "Updated description" }

&nbsp; }

);

console.log(updatedEval);

Response

{

&nbsp; "object": "eval",

&nbsp; "id": "eval\_67abd54d9b0081909a86353f6fb9317a",

&nbsp; "data\_source\_config": {

&nbsp;   "type": "custom",

&nbsp;   "schema": {

&nbsp;     "type": "object",

&nbsp;     "properties": {

&nbsp;       "item": {

&nbsp;         "type": "object",

&nbsp;         "properties": {

&nbsp;           "input": {

&nbsp;             "type": "string"

&nbsp;           },

&nbsp;           "ground\_truth": {

&nbsp;             "type": "string"

&nbsp;           }

&nbsp;         },

&nbsp;         "required": \[

&nbsp;           "input",

&nbsp;           "ground\_truth"

&nbsp;         ]

&nbsp;       }

&nbsp;     },

&nbsp;     "required": \[

&nbsp;       "item"

&nbsp;     ]

&nbsp;   }

&nbsp; },

&nbsp; "testing\_criteria": \[

&nbsp;   {

&nbsp;     "name": "String check",

&nbsp;     "id": "String check-2eaf2d8d-d649-4335-8148-9535a7ca73c2",

&nbsp;     "type": "string\_check",

&nbsp;     "input": "{{item.input}}",

&nbsp;     "reference": "{{item.ground\_truth}}",

&nbsp;     "operation": "eq"

&nbsp;   }

&nbsp; ],

&nbsp; "name": "Updated Eval",

&nbsp; "created\_at": 1739314509,

&nbsp; "metadata": {"description": "Updated description"},

}

Delete an eval

delete

&nbsp;

https://api.openai.com/v1/evals/{eval\_id}

Delete an evaluation.



Path parameters

eval\_id

string



Required

The ID of the evaluation to delete.



Returns

A deletion confirmation object.



Example request

import OpenAI from "openai";



const openai = new OpenAI();



const deleted = await openai.evals.delete("eval\_abc123");

console.log(deleted);

Response

{

&nbsp; "object": "eval.deleted",

&nbsp; "deleted": true,

&nbsp; "eval\_id": "eval\_abc123"

}

List evals

get

&nbsp;

https://api.openai.com/v1/evals

List evaluations for a project.



Query parameters

after

string



Optional

Identifier for the last eval from the previous pagination request.



limit

integer



Optional

Defaults to 20

Number of evals to retrieve.



order

string



Optional

Defaults to asc

Sort order for evals by timestamp. Use asc for ascending order or desc for descending order.



order\_by

string



Optional

Defaults to created\_at

Evals can be ordered by creation time or last updated time. Use created\_at for creation time or updated\_at for last updated time.



Returns

A list of evals matching the specified filters.



Example request

import OpenAI from "openai";



const openai = new OpenAI();



const evals = await openai.evals.list({ limit: 1 });

console.log(evals);

Response

{

&nbsp; "object": "list",

&nbsp; "data": \[

&nbsp;   {

&nbsp;     "id": "eval\_67abd54d9b0081909a86353f6fb9317a",

&nbsp;     "object": "eval",

&nbsp;     "data\_source\_config": {

&nbsp;       "type": "stored\_completions",

&nbsp;       "metadata": {

&nbsp;         "usecase": "push\_notifications\_summarizer"

&nbsp;       },

&nbsp;       "schema": {

&nbsp;         "type": "object",

&nbsp;         "properties": {

&nbsp;           "item": {

&nbsp;             "type": "object"

&nbsp;           },

&nbsp;           "sample": {

&nbsp;             "type": "object"

&nbsp;           }

&nbsp;         },

&nbsp;         "required": \[

&nbsp;           "item",

&nbsp;           "sample"

&nbsp;         ]

&nbsp;       }

&nbsp;     },

&nbsp;     "testing\_criteria": \[

&nbsp;       {

&nbsp;         "name": "Push Notification Summary Grader",

&nbsp;         "id": "Push Notification Summary Grader-9b876f24-4762-4be9-aff4-db7a9b31c673",

&nbsp;         "type": "label\_model",

&nbsp;         "model": "o3-mini",

&nbsp;         "input": \[

&nbsp;           {

&nbsp;             "type": "message",

&nbsp;             "role": "developer",

&nbsp;             "content": {

&nbsp;               "type": "input\_text",

&nbsp;               "text": "\\nLabel the following push notification summary as either correct or incorrect.\\nThe push notification and the summary will be provided below.\\nA good push notificiation summary is concise and snappy.\\nIf it is good, then label it as correct, if not, then incorrect.\\n"

&nbsp;             }

&nbsp;           },

&nbsp;           {

&nbsp;             "type": "message",

&nbsp;             "role": "user",

&nbsp;             "content": {

&nbsp;               "type": "input\_text",

&nbsp;               "text": "\\nPush notifications: {{item.input}}\\nSummary: {{sample.output\_text}}\\n"

&nbsp;             }

&nbsp;           }

&nbsp;         ],

&nbsp;         "passing\_labels": \[

&nbsp;           "correct"

&nbsp;         ],

&nbsp;         "labels": \[

&nbsp;           "correct",

&nbsp;           "incorrect"

&nbsp;         ],

&nbsp;         "sampling\_params": null

&nbsp;       }

&nbsp;     ],

&nbsp;     "name": "Push Notification Summary Grader",

&nbsp;     "created\_at": 1739314509,

&nbsp;     "metadata": {

&nbsp;       "description": "A stored completions eval for push notification summaries"

&nbsp;     }

&nbsp;   }

&nbsp; ],

&nbsp; "first\_id": "eval\_67abd54d9b0081909a86353f6fb9317a",

&nbsp; "last\_id": "eval\_67aa884cf6688190b58f657d4441c8b7",

&nbsp; "has\_more": true

}

Get eval runs

get

&nbsp;

https://api.openai.com/v1/evals/{eval\_id}/runs

Get a list of runs for an evaluation.



Path parameters

eval\_id

string



Required

The ID of the evaluation to retrieve runs for.



Query parameters

after

string



Optional

Identifier for the last run from the previous pagination request.



limit

integer



Optional

Defaults to 20

Number of runs to retrieve.



order

string



Optional

Defaults to asc

Sort order for runs by timestamp. Use asc for ascending order or desc for descending order. Defaults to asc.



status

string



Optional

Filter runs by status. One of queued | in\_progress | failed | completed | canceled.



Returns

A list of EvalRun objects matching the specified ID.



Example request

import OpenAI from "openai";



const openai = new OpenAI();



const runs = await openai.evals.runs.list("egroup\_67abd54d9b0081909a86353f6fb9317a");

console.log(runs);

Response

{

&nbsp; "object": "list",

&nbsp; "data": \[

&nbsp;   {

&nbsp;     "object": "eval.run",

&nbsp;     "id": "evalrun\_67e0c7d31560819090d60c0780591042",

&nbsp;     "eval\_id": "eval\_67e0c726d560819083f19a957c4c640b",

&nbsp;     "report\_url": "https://platform.openai.com/evaluations/eval\_67e0c726d560819083f19a957c4c640b",

&nbsp;     "status": "completed",

&nbsp;     "model": "o3-mini",

&nbsp;     "name": "bulk\_with\_negative\_examples\_o3-mini",

&nbsp;     "created\_at": 1742784467,

&nbsp;     "result\_counts": {

&nbsp;       "total": 1,

&nbsp;       "errored": 0,

&nbsp;       "failed": 0,

&nbsp;       "passed": 1

&nbsp;     },

&nbsp;     "per\_model\_usage": \[

&nbsp;       {

&nbsp;         "model\_name": "o3-mini",

&nbsp;         "invocation\_count": 1,

&nbsp;         "prompt\_tokens": 563,

&nbsp;         "completion\_tokens": 874,

&nbsp;         "total\_tokens": 1437,

&nbsp;         "cached\_tokens": 0

&nbsp;       }

&nbsp;     ],

&nbsp;     "per\_testing\_criteria\_results": \[

&nbsp;       {

&nbsp;         "testing\_criteria": "Push Notification Summary Grader-1808cd0b-eeec-4e0b-a519-337e79f4f5d1",

&nbsp;         "passed": 1,

&nbsp;         "failed": 0

&nbsp;       }

&nbsp;     ],

&nbsp;     "data\_source": {

&nbsp;       "type": "completions",

&nbsp;       "source": {

&nbsp;         "type": "file\_content",

&nbsp;         "content": \[

&nbsp;           {

&nbsp;             "item": {

&nbsp;               "notifications": "\\n- New message from Sarah: \\"Can you call me later?\\"\\n- Your package has been delivered!\\n- Flash sale: 20% off electronics for the next 2 hours!\\n"

&nbsp;             }

&nbsp;           }

&nbsp;         ]

&nbsp;       },

&nbsp;       "input\_messages": {

&nbsp;         "type": "template",

&nbsp;         "template": \[

&nbsp;           {

&nbsp;             "type": "message",

&nbsp;             "role": "developer",

&nbsp;             "content": {

&nbsp;               "type": "input\_text",

&nbsp;               "text": "\\n\\n\\n\\nYou are a helpful assistant that takes in an array of push notifications and returns a collapsed summary of them.\\nThe push notification will be provided as follows:\\n<push\_notifications>\\n...notificationlist...\\n</push\_notifications>\\n\\nYou should return just the summary and nothing else.\\n\\n\\nYou should return a summary that is concise and snappy.\\n\\n\\nHere is an example of a good summary:\\n<push\_notifications>\\n- Traffic alert: Accident reported on Main Street.- Package out for delivery: Expected by 5 PM.- New friend suggestion: Connect with Emma.\\n</push\_notifications>\\n<summary>\\nTraffic alert, package expected by 5pm, suggestion for new friend (Emily).\\n</summary>\\n\\n\\nHere is an example of a bad summary:\\n<push\_notifications>\\n- Traffic alert: Accident reported on Main Street.- Package out for delivery: Expected by 5 PM.- New friend suggestion: Connect with Emma.\\n</push\_notifications>\\n<summary>\\nTraffic alert reported on main street. You have a package that will arrive by 5pm, Emily is a new friend suggested for you.\\n</summary>\\n"

&nbsp;             }

&nbsp;           },

&nbsp;           {

&nbsp;             "type": "message",

&nbsp;             "role": "user",

&nbsp;             "content": {

&nbsp;               "type": "input\_text",

&nbsp;               "text": "<push\_notifications>{{item.notifications}}</push\_notifications>"

&nbsp;             }

&nbsp;           }

&nbsp;         ]

&nbsp;       },

&nbsp;       "model": "o3-mini",

&nbsp;       "sampling\_params": null

&nbsp;     },

&nbsp;     "error": null,

&nbsp;     "metadata": {}

&nbsp;   }

&nbsp; ],

&nbsp; "first\_id": "evalrun\_67e0c7d31560819090d60c0780591042",

&nbsp; "last\_id": "evalrun\_67e0c7d31560819090d60c0780591042",

&nbsp; "has\_more": true

}

Get an eval run

get

&nbsp;

https://api.openai.com/v1/evals/{eval\_id}/runs/{run\_id}

Get an evaluation run by ID.



Path parameters

eval\_id

string



Required

The ID of the evaluation to retrieve runs for.



run\_id

string



Required

The ID of the run to retrieve.



Returns

The EvalRun object matching the specified ID.



Example request

import OpenAI from "openai";



const openai = new OpenAI();



const run = await openai.evals.runs.retrieve(

&nbsp; "evalrun\_67abd54d60ec8190832b46859da808f7",

&nbsp; { eval\_id: "eval\_67abd54d9b0081909a86353f6fb9317a" }

);

console.log(run);

Response

{

&nbsp; "object": "eval.run",

&nbsp; "id": "evalrun\_67abd54d60ec8190832b46859da808f7",

&nbsp; "eval\_id": "eval\_67abd54d9b0081909a86353f6fb9317a",

&nbsp; "report\_url": "https://platform.openai.com/evaluations/eval\_67abd54d9b0081909a86353f6fb9317a?run\_id=evalrun\_67abd54d60ec8190832b46859da808f7",

&nbsp; "status": "queued",

&nbsp; "model": "gpt-4o-mini",

&nbsp; "name": "gpt-4o-mini",

&nbsp; "created\_at": 1743092069,

&nbsp; "result\_counts": {

&nbsp;   "total": 0,

&nbsp;   "errored": 0,

&nbsp;   "failed": 0,

&nbsp;   "passed": 0

&nbsp; },

&nbsp; "per\_model\_usage": null,

&nbsp; "per\_testing\_criteria\_results": null,

&nbsp; "data\_source": {

&nbsp;   "type": "completions",

&nbsp;   "source": {

&nbsp;     "type": "file\_content",

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Tech Company Launches Advanced Artificial Intelligence Platform",

&nbsp;           "ground\_truth": "Technology"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Central Bank Increases Interest Rates Amid Inflation Concerns",

&nbsp;           "ground\_truth": "Markets"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "International Summit Addresses Climate Change Strategies",

&nbsp;           "ground\_truth": "World"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Major Retailer Reports Record-Breaking Holiday Sales",

&nbsp;           "ground\_truth": "Business"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "National Team Qualifies for World Championship Finals",

&nbsp;           "ground\_truth": "Sports"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Stock Markets Rally After Positive Economic Data Released",

&nbsp;           "ground\_truth": "Markets"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Global Manufacturer Announces Merger with Competitor",

&nbsp;           "ground\_truth": "Business"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Breakthrough in Renewable Energy Technology Unveiled",

&nbsp;           "ground\_truth": "Technology"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "World Leaders Sign Historic Climate Agreement",

&nbsp;           "ground\_truth": "World"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Professional Athlete Sets New Record in Championship Event",

&nbsp;           "ground\_truth": "Sports"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Financial Institutions Adapt to New Regulatory Requirements",

&nbsp;           "ground\_truth": "Business"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Tech Conference Showcases Advances in Artificial Intelligence",

&nbsp;           "ground\_truth": "Technology"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Global Markets Respond to Oil Price Fluctuations",

&nbsp;           "ground\_truth": "Markets"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "International Cooperation Strengthened Through New Treaty",

&nbsp;           "ground\_truth": "World"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Sports League Announces Revised Schedule for Upcoming Season",

&nbsp;           "ground\_truth": "Sports"

&nbsp;         }

&nbsp;       }

&nbsp;     ]

&nbsp;   },

&nbsp;   "input\_messages": {

&nbsp;     "type": "template",

&nbsp;     "template": \[

&nbsp;       {

&nbsp;         "type": "message",

&nbsp;         "role": "developer",

&nbsp;         "content": {

&nbsp;           "type": "input\_text",

&nbsp;           "text": "Categorize a given news headline into one of the following topics: Technology, Markets, World, Business, or Sports.\\n\\n# Steps\\n\\n1. Analyze the content of the news headline to understand its primary focus.\\n2. Extract the subject matter, identifying any key indicators or keywords.\\n3. Use the identified indicators to determine the most suitable category out of the five options: Technology, Markets, World, Business, or Sports.\\n4. Ensure only one category is selected per headline.\\n\\n# Output Format\\n\\nRespond with the chosen category as a single word. For instance: \\"Technology\\", \\"Markets\\", \\"World\\", \\"Business\\", or \\"Sports\\".\\n\\n# Examples\\n\\n\*\*Input\*\*: \\"Apple Unveils New iPhone Model, Featuring Advanced AI Features\\"  \\n\*\*Output\*\*: \\"Technology\\"\\n\\n\*\*Input\*\*: \\"Global Stocks Mixed as Investors Await Central Bank Decisions\\"  \\n\*\*Output\*\*: \\"Markets\\"\\n\\n\*\*Input\*\*: \\"War in Ukraine: Latest Updates on Negotiation Status\\"  \\n\*\*Output\*\*: \\"World\\"\\n\\n\*\*Input\*\*: \\"Microsoft in Talks to Acquire Gaming Company for $2 Billion\\"  \\n\*\*Output\*\*: \\"Business\\"\\n\\n\*\*Input\*\*: \\"Manchester United Secures Win in Premier League Football Match\\"  \\n\*\*Output\*\*: \\"Sports\\" \\n\\n# Notes\\n\\n- If the headline appears to fit into more than one category, choose the most dominant theme.\\n- Keywords or phrases such as \\"stocks\\", \\"company acquisition\\", \\"match\\", or technological brands can be good indicators for classification.\\n"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "type": "message",

&nbsp;         "role": "user",

&nbsp;         "content": {

&nbsp;           "type": "input\_text",

&nbsp;           "text": "{{item.input}}"

&nbsp;         }

&nbsp;       }

&nbsp;     ]

&nbsp;   },

&nbsp;   "model": "gpt-4o-mini",

&nbsp;   "sampling\_params": {

&nbsp;     "seed": 42,

&nbsp;     "temperature": 1.0,

&nbsp;     "top\_p": 1.0,

&nbsp;     "max\_completions\_tokens": 2048

&nbsp;   }

&nbsp; },

&nbsp; "error": null,

&nbsp; "metadata": {}

}

Create eval run

post

&nbsp;

https://api.openai.com/v1/evals/{eval\_id}/runs

Kicks off a new run for a given evaluation, specifying the data source, and what model configuration to use to test. The datasource will be validated against the schema specified in the config of the evaluation.



Path parameters

eval\_id

string



Required

The ID of the evaluation to create a run for.



Request body

data\_source

object



Required

Details about the run's data source.





Show possible types

metadata

map



Optional

Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard.



Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.



name

string



Optional

The name of the run.



Returns

The EvalRun object matching the specified ID.



Example request

import OpenAI from "openai";



const openai = new OpenAI();



const run = await openai.evals.runs.create(

&nbsp; "eval\_67e579652b548190aaa83ada4b125f47",

&nbsp; {

&nbsp;   name: "gpt-4o-mini",

&nbsp;   data\_source: {

&nbsp;     type: "completions",

&nbsp;     input\_messages: {

&nbsp;       type: "template",

&nbsp;       template: \[

&nbsp;         {

&nbsp;           role: "developer",

&nbsp;           content: "Categorize a given news headline into one of the following topics: Technology, Markets, World, Business, or Sports.\\n\\n# Steps\\n\\n1. Analyze the content of the news headline to understand its primary focus.\\n2. Extract the subject matter, identifying any key indicators or keywords.\\n3. Use the identified indicators to determine the most suitable category out of the five options: Technology, Markets, World, Business, or Sports.\\n4. Ensure only one category is selected per headline.\\n\\n# Output Format\\n\\nRespond with the chosen category as a single word. For instance: \\"Technology\\", \\"Markets\\", \\"World\\", \\"Business\\", or \\"Sports\\".\\n\\n# Examples\\n\\n\*\*Input\*\*: \\"Apple Unveils New iPhone Model, Featuring Advanced AI Features\\"  \\n\*\*Output\*\*: \\"Technology\\"\\n\\n\*\*Input\*\*: \\"Global Stocks Mixed as Investors Await Central Bank Decisions\\"  \\n\*\*Output\*\*: \\"Markets\\"\\n\\n\*\*Input\*\*: \\"War in Ukraine: Latest Updates on Negotiation Status\\"  \\n\*\*Output\*\*: \\"World\\"\\n\\n\*\*Input\*\*: \\"Microsoft in Talks to Acquire Gaming Company for $2 Billion\\"  \\n\*\*Output\*\*: \\"Business\\"\\n\\n\*\*Input\*\*: \\"Manchester United Secures Win in Premier League Football Match\\"  \\n\*\*Output\*\*: \\"Sports\\" \\n\\n# Notes\\n\\n- If the headline appears to fit into more than one category, choose the most dominant theme.\\n- Keywords or phrases such as \\"stocks\\", \\"company acquisition\\", \\"match\\", or technological brands can be good indicators for classification.\\n"

&nbsp;         },

&nbsp;         {

&nbsp;           role: "user",

&nbsp;           content: "{{item.input}}"

&nbsp;         }

&nbsp;       ]

&nbsp;     },

&nbsp;     sampling\_params: {

&nbsp;       temperature: 1,

&nbsp;       max\_completions\_tokens: 2048,

&nbsp;       top\_p: 1,

&nbsp;       seed: 42

&nbsp;     },

&nbsp;     model: "gpt-4o-mini",

&nbsp;     source: {

&nbsp;       type: "file\_content",

&nbsp;       content: \[

&nbsp;         {

&nbsp;           item: {

&nbsp;             input: "Tech Company Launches Advanced Artificial Intelligence Platform",

&nbsp;             ground\_truth: "Technology"

&nbsp;           }

&nbsp;         }

&nbsp;       ]

&nbsp;     }

&nbsp;   }

&nbsp; }

);

console.log(run);

Response

{

&nbsp; "object": "eval.run",

&nbsp; "id": "evalrun\_67e57965b480819094274e3a32235e4c",

&nbsp; "eval\_id": "eval\_67e579652b548190aaa83ada4b125f47",

&nbsp; "report\_url": "https://platform.openai.com/evaluations/eval\_67e579652b548190aaa83ada4b125f47\&run\_id=evalrun\_67e57965b480819094274e3a32235e4c",

&nbsp; "status": "queued",

&nbsp; "model": "gpt-4o-mini",

&nbsp; "name": "gpt-4o-mini",

&nbsp; "created\_at": 1743092069,

&nbsp; "result\_counts": {

&nbsp;   "total": 0,

&nbsp;   "errored": 0,

&nbsp;   "failed": 0,

&nbsp;   "passed": 0

&nbsp; },

&nbsp; "per\_model\_usage": null,

&nbsp; "per\_testing\_criteria\_results": null,

&nbsp; "data\_source": {

&nbsp;   "type": "completions",

&nbsp;   "source": {

&nbsp;     "type": "file\_content",

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Tech Company Launches Advanced Artificial Intelligence Platform",

&nbsp;           "ground\_truth": "Technology"

&nbsp;         }

&nbsp;       }

&nbsp;     ]

&nbsp;   },

&nbsp;   "input\_messages": {

&nbsp;     "type": "template",

&nbsp;     "template": \[

&nbsp;       {

&nbsp;         "type": "message",

&nbsp;         "role": "developer",

&nbsp;         "content": {

&nbsp;           "type": "input\_text",

&nbsp;           "text": "Categorize a given news headline into one of the following topics: Technology, Markets, World, Business, or Sports.\\n\\n# Steps\\n\\n1. Analyze the content of the news headline to understand its primary focus.\\n2. Extract the subject matter, identifying any key indicators or keywords.\\n3. Use the identified indicators to determine the most suitable category out of the five options: Technology, Markets, World, Business, or Sports.\\n4. Ensure only one category is selected per headline.\\n\\n# Output Format\\n\\nRespond with the chosen category as a single word. For instance: \\"Technology\\", \\"Markets\\", \\"World\\", \\"Business\\", or \\"Sports\\".\\n\\n# Examples\\n\\n\*\*Input\*\*: \\"Apple Unveils New iPhone Model, Featuring Advanced AI Features\\"  \\n\*\*Output\*\*: \\"Technology\\"\\n\\n\*\*Input\*\*: \\"Global Stocks Mixed as Investors Await Central Bank Decisions\\"  \\n\*\*Output\*\*: \\"Markets\\"\\n\\n\*\*Input\*\*: \\"War in Ukraine: Latest Updates on Negotiation Status\\"  \\n\*\*Output\*\*: \\"World\\"\\n\\n\*\*Input\*\*: \\"Microsoft in Talks to Acquire Gaming Company for $2 Billion\\"  \\n\*\*Output\*\*: \\"Business\\"\\n\\n\*\*Input\*\*: \\"Manchester United Secures Win in Premier League Football Match\\"  \\n\*\*Output\*\*: \\"Sports\\" \\n\\n# Notes\\n\\n- If the headline appears to fit into more than one category, choose the most dominant theme.\\n- Keywords or phrases such as \\"stocks\\", \\"company acquisition\\", \\"match\\", or technological brands can be good indicators for classification.\\n"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "type": "message",

&nbsp;         "role": "user",

&nbsp;         "content": {

&nbsp;           "type": "input\_text",

&nbsp;           "text": "{{item.input}}"

&nbsp;         }

&nbsp;       }

&nbsp;     ]

&nbsp;   },

&nbsp;   "model": "gpt-4o-mini",

&nbsp;   "sampling\_params": {

&nbsp;     "seed": 42,

&nbsp;     "temperature": 1.0,

&nbsp;     "top\_p": 1.0,

&nbsp;     "max\_completions\_tokens": 2048

&nbsp;   }

&nbsp; },

&nbsp; "error": null,

&nbsp; "metadata": {}

}

Cancel eval run

post

&nbsp;

https://api.openai.com/v1/evals/{eval\_id}/runs/{run\_id}

Cancel an ongoing evaluation run.



Path parameters

eval\_id

string



Required

The ID of the evaluation whose run you want to cancel.



run\_id

string



Required

The ID of the run to cancel.



Returns

The updated EvalRun object reflecting that the run is canceled.



Example request

import OpenAI from "openai";



const openai = new OpenAI();



const canceledRun = await openai.evals.runs.cancel(

&nbsp; "evalrun\_67abd54d60ec8190832b46859da808f7",

&nbsp; { eval\_id: "eval\_67abd54d9b0081909a86353f6fb9317a" }

);

console.log(canceledRun);

Response

{

&nbsp; "object": "eval.run",

&nbsp; "id": "evalrun\_67abd54d60ec8190832b46859da808f7",

&nbsp; "eval\_id": "eval\_67abd54d9b0081909a86353f6fb9317a",

&nbsp; "report\_url": "https://platform.openai.com/evaluations/eval\_67abd54d9b0081909a86353f6fb9317a?run\_id=evalrun\_67abd54d60ec8190832b46859da808f7",

&nbsp; "status": "canceled",

&nbsp; "model": "gpt-4o-mini",

&nbsp; "name": "gpt-4o-mini",

&nbsp; "created\_at": 1743092069,

&nbsp; "result\_counts": {

&nbsp;   "total": 0,

&nbsp;   "errored": 0,

&nbsp;   "failed": 0,

&nbsp;   "passed": 0

&nbsp; },

&nbsp; "per\_model\_usage": null,

&nbsp; "per\_testing\_criteria\_results": null,

&nbsp; "data\_source": {

&nbsp;   "type": "completions",

&nbsp;   "source": {

&nbsp;     "type": "file\_content",

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Tech Company Launches Advanced Artificial Intelligence Platform",

&nbsp;           "ground\_truth": "Technology"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Central Bank Increases Interest Rates Amid Inflation Concerns",

&nbsp;           "ground\_truth": "Markets"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "International Summit Addresses Climate Change Strategies",

&nbsp;           "ground\_truth": "World"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Major Retailer Reports Record-Breaking Holiday Sales",

&nbsp;           "ground\_truth": "Business"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "National Team Qualifies for World Championship Finals",

&nbsp;           "ground\_truth": "Sports"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Stock Markets Rally After Positive Economic Data Released",

&nbsp;           "ground\_truth": "Markets"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Global Manufacturer Announces Merger with Competitor",

&nbsp;           "ground\_truth": "Business"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Breakthrough in Renewable Energy Technology Unveiled",

&nbsp;           "ground\_truth": "Technology"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "World Leaders Sign Historic Climate Agreement",

&nbsp;           "ground\_truth": "World"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Professional Athlete Sets New Record in Championship Event",

&nbsp;           "ground\_truth": "Sports"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Financial Institutions Adapt to New Regulatory Requirements",

&nbsp;           "ground\_truth": "Business"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Tech Conference Showcases Advances in Artificial Intelligence",

&nbsp;           "ground\_truth": "Technology"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Global Markets Respond to Oil Price Fluctuations",

&nbsp;           "ground\_truth": "Markets"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "International Cooperation Strengthened Through New Treaty",

&nbsp;           "ground\_truth": "World"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Sports League Announces Revised Schedule for Upcoming Season",

&nbsp;           "ground\_truth": "Sports"

&nbsp;         }

&nbsp;       }

&nbsp;     ]

&nbsp;   },

&nbsp;   "input\_messages": {

&nbsp;     "type": "template",

&nbsp;     "template": \[

&nbsp;       {

&nbsp;         "type": "message",

&nbsp;         "role": "developer",

&nbsp;         "content": {

&nbsp;           "type": "input\_text",

&nbsp;           "text": "Categorize a given news headline into one of the following topics: Technology, Markets, World, Business, or Sports.\\n\\n# Steps\\n\\n1. Analyze the content of the news headline to understand its primary focus.\\n2. Extract the subject matter, identifying any key indicators or keywords.\\n3. Use the identified indicators to determine the most suitable category out of the five options: Technology, Markets, World, Business, or Sports.\\n4. Ensure only one category is selected per headline.\\n\\n# Output Format\\n\\nRespond with the chosen category as a single word. For instance: \\"Technology\\", \\"Markets\\", \\"World\\", \\"Business\\", or \\"Sports\\".\\n\\n# Examples\\n\\n\*\*Input\*\*: \\"Apple Unveils New iPhone Model, Featuring Advanced AI Features\\"  \\n\*\*Output\*\*: \\"Technology\\"\\n\\n\*\*Input\*\*: \\"Global Stocks Mixed as Investors Await Central Bank Decisions\\"  \\n\*\*Output\*\*: \\"Markets\\"\\n\\n\*\*Input\*\*: \\"War in Ukraine: Latest Updates on Negotiation Status\\"  \\n\*\*Output\*\*: \\"World\\"\\n\\n\*\*Input\*\*: \\"Microsoft in Talks to Acquire Gaming Company for $2 Billion\\"  \\n\*\*Output\*\*: \\"Business\\"\\n\\n\*\*Input\*\*: \\"Manchester United Secures Win in Premier League Football Match\\"  \\n\*\*Output\*\*: \\"Sports\\" \\n\\n# Notes\\n\\n- If the headline appears to fit into more than one category, choose the most dominant theme.\\n- Keywords or phrases such as \\"stocks\\", \\"company acquisition\\", \\"match\\", or technological brands can be good indicators for classification.\\n"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "type": "message",

&nbsp;         "role": "user",

&nbsp;         "content": {

&nbsp;           "type": "input\_text",

&nbsp;           "text": "{{item.input}}"

&nbsp;         }

&nbsp;       }

&nbsp;     ]

&nbsp;   },

&nbsp;   "model": "gpt-4o-mini",

&nbsp;   "sampling\_params": {

&nbsp;     "seed": 42,

&nbsp;     "temperature": 1.0,

&nbsp;     "top\_p": 1.0,

&nbsp;     "max\_completions\_tokens": 2048

&nbsp;   }

&nbsp; },

&nbsp; "error": null,

&nbsp; "metadata": {}

}

Delete eval run

delete

&nbsp;

https://api.openai.com/v1/evals/{eval\_id}/runs/{run\_id}

Delete an eval run.



Path parameters

eval\_id

string



Required

The ID of the evaluation to delete the run from.



run\_id

string



Required

The ID of the run to delete.



Returns

An object containing the status of the delete operation.



Example request

import OpenAI from "openai";



const openai = new OpenAI();



const deleted = await openai.evals.runs.delete(

&nbsp; "eval\_123abc",

&nbsp; "evalrun\_abc456"

);

console.log(deleted);

Response

{

&nbsp; "object": "eval.run.deleted",

&nbsp; "deleted": true,

&nbsp; "run\_id": "evalrun\_abc456"

}

Get an output item of an eval run

get

&nbsp;

https://api.openai.com/v1/evals/{eval\_id}/runs/{run\_id}/output\_items/{output\_item\_id}

Get an evaluation run output item by ID.



Path parameters

eval\_id

string



Required

The ID of the evaluation to retrieve runs for.



output\_item\_id

string



Required

The ID of the output item to retrieve.



run\_id

string



Required

The ID of the run to retrieve.



Returns

The EvalRunOutputItem object matching the specified ID.



Example request

import OpenAI from "openai";



const openai = new OpenAI();



const outputItem = await openai.evals.runs.outputItems.retrieve(

&nbsp; "outputitem\_67abd55eb6548190bb580745d5644a33",

&nbsp; {

&nbsp;   eval\_id: "eval\_67abd54d9b0081909a86353f6fb9317a",

&nbsp;   run\_id: "evalrun\_67abd54d60ec8190832b46859da808f7",

&nbsp; }

);

console.log(outputItem);

Response

{

&nbsp; "object": "eval.run.output\_item",

&nbsp; "id": "outputitem\_67e5796c28e081909917bf79f6e6214d",

&nbsp; "created\_at": 1743092076,

&nbsp; "run\_id": "evalrun\_67abd54d60ec8190832b46859da808f7",

&nbsp; "eval\_id": "eval\_67abd54d9b0081909a86353f6fb9317a",

&nbsp; "status": "pass",

&nbsp; "datasource\_item\_id": 5,

&nbsp; "datasource\_item": {

&nbsp;   "input": "Stock Markets Rally After Positive Economic Data Released",

&nbsp;   "ground\_truth": "Markets"

&nbsp; },

&nbsp; "results": \[

&nbsp;   {

&nbsp;     "name": "String check-a2486074-d803-4445-b431-ad2262e85d47",

&nbsp;     "sample": null,

&nbsp;     "passed": true,

&nbsp;     "score": 1.0

&nbsp;   }

&nbsp; ],

&nbsp; "sample": {

&nbsp;   "input": \[

&nbsp;     {

&nbsp;       "role": "developer",

&nbsp;       "content": "Categorize a given news headline into one of the following topics: Technology, Markets, World, Business, or Sports.\\n\\n# Steps\\n\\n1. Analyze the content of the news headline to understand its primary focus.\\n2. Extract the subject matter, identifying any key indicators or keywords.\\n3. Use the identified indicators to determine the most suitable category out of the five options: Technology, Markets, World, Business, or Sports.\\n4. Ensure only one category is selected per headline.\\n\\n# Output Format\\n\\nRespond with the chosen category as a single word. For instance: \\"Technology\\", \\"Markets\\", \\"World\\", \\"Business\\", or \\"Sports\\".\\n\\n# Examples\\n\\n\*\*Input\*\*: \\"Apple Unveils New iPhone Model, Featuring Advanced AI Features\\"  \\n\*\*Output\*\*: \\"Technology\\"\\n\\n\*\*Input\*\*: \\"Global Stocks Mixed as Investors Await Central Bank Decisions\\"  \\n\*\*Output\*\*: \\"Markets\\"\\n\\n\*\*Input\*\*: \\"War in Ukraine: Latest Updates on Negotiation Status\\"  \\n\*\*Output\*\*: \\"World\\"\\n\\n\*\*Input\*\*: \\"Microsoft in Talks to Acquire Gaming Company for $2 Billion\\"  \\n\*\*Output\*\*: \\"Business\\"\\n\\n\*\*Input\*\*: \\"Manchester United Secures Win in Premier League Football Match\\"  \\n\*\*Output\*\*: \\"Sports\\" \\n\\n# Notes\\n\\n- If the headline appears to fit into more than one category, choose the most dominant theme.\\n- Keywords or phrases such as \\"stocks\\", \\"company acquisition\\", \\"match\\", or technological brands can be good indicators for classification.\\n",

&nbsp;       "tool\_call\_id": null,

&nbsp;       "tool\_calls": null,

&nbsp;       "function\_call": null

&nbsp;     },

&nbsp;     {

&nbsp;       "role": "user",

&nbsp;       "content": "Stock Markets Rally After Positive Economic Data Released",

&nbsp;       "tool\_call\_id": null,

&nbsp;       "tool\_calls": null,

&nbsp;       "function\_call": null

&nbsp;     }

&nbsp;   ],

&nbsp;   "output": \[

&nbsp;     {

&nbsp;       "role": "assistant",

&nbsp;       "content": "Markets",

&nbsp;       "tool\_call\_id": null,

&nbsp;       "tool\_calls": null,

&nbsp;       "function\_call": null

&nbsp;     }

&nbsp;   ],

&nbsp;   "finish\_reason": "stop",

&nbsp;   "model": "gpt-4o-mini-2024-07-18",

&nbsp;   "usage": {

&nbsp;     "total\_tokens": 325,

&nbsp;     "completion\_tokens": 2,

&nbsp;     "prompt\_tokens": 323,

&nbsp;     "cached\_tokens": 0

&nbsp;   },

&nbsp;   "error": null,

&nbsp;   "temperature": 1.0,

&nbsp;   "max\_completion\_tokens": 2048,

&nbsp;   "top\_p": 1.0,

&nbsp;   "seed": 42

&nbsp; }

}

Get eval run output items

get

&nbsp;

https://api.openai.com/v1/evals/{eval\_id}/runs/{run\_id}/output\_items

Get a list of output items for an evaluation run.



Path parameters

eval\_id

string



Required

The ID of the evaluation to retrieve runs for.



run\_id

string



Required

The ID of the run to retrieve output items for.



Query parameters

after

string



Optional

Identifier for the last output item from the previous pagination request.



limit

integer



Optional

Defaults to 20

Number of output items to retrieve.



order

string



Optional

Defaults to asc

Sort order for output items by timestamp. Use asc for ascending order or desc for descending order. Defaults to asc.



status

string



Optional

Filter output items by status. Use failed to filter by failed output items or pass to filter by passed output items.



Returns

A list of EvalRunOutputItem objects matching the specified ID.



Example request

import OpenAI from "openai";



const openai = new OpenAI();



const outputItems = await openai.evals.runs.outputItems.list(

&nbsp; "egroup\_67abd54d9b0081909a86353f6fb9317a",

&nbsp; "erun\_67abd54d60ec8190832b46859da808f7"

);

console.log(outputItems);

Response

{

&nbsp; "object": "list",

&nbsp; "data": \[

&nbsp;   {

&nbsp;     "object": "eval.run.output\_item",

&nbsp;     "id": "outputitem\_67e5796c28e081909917bf79f6e6214d",

&nbsp;     "created\_at": 1743092076,

&nbsp;     "run\_id": "evalrun\_67abd54d60ec8190832b46859da808f7",

&nbsp;     "eval\_id": "eval\_67abd54d9b0081909a86353f6fb9317a",

&nbsp;     "status": "pass",

&nbsp;     "datasource\_item\_id": 5,

&nbsp;     "datasource\_item": {

&nbsp;       "input": "Stock Markets Rally After Positive Economic Data Released",

&nbsp;       "ground\_truth": "Markets"

&nbsp;     },

&nbsp;     "results": \[

&nbsp;       {

&nbsp;         "name": "String check-a2486074-d803-4445-b431-ad2262e85d47",

&nbsp;         "sample": null,

&nbsp;         "passed": true,

&nbsp;         "score": 1.0

&nbsp;       }

&nbsp;     ],

&nbsp;     "sample": {

&nbsp;       "input": \[

&nbsp;         {

&nbsp;           "role": "developer",

&nbsp;           "content": "Categorize a given news headline into one of the following topics: Technology, Markets, World, Business, or Sports.\\n\\n# Steps\\n\\n1. Analyze the content of the news headline to understand its primary focus.\\n2. Extract the subject matter, identifying any key indicators or keywords.\\n3. Use the identified indicators to determine the most suitable category out of the five options: Technology, Markets, World, Business, or Sports.\\n4. Ensure only one category is selected per headline.\\n\\n# Output Format\\n\\nRespond with the chosen category as a single word. For instance: \\"Technology\\", \\"Markets\\", \\"World\\", \\"Business\\", or \\"Sports\\".\\n\\n# Examples\\n\\n\*\*Input\*\*: \\"Apple Unveils New iPhone Model, Featuring Advanced AI Features\\"  \\n\*\*Output\*\*: \\"Technology\\"\\n\\n\*\*Input\*\*: \\"Global Stocks Mixed as Investors Await Central Bank Decisions\\"  \\n\*\*Output\*\*: \\"Markets\\"\\n\\n\*\*Input\*\*: \\"War in Ukraine: Latest Updates on Negotiation Status\\"  \\n\*\*Output\*\*: \\"World\\"\\n\\n\*\*Input\*\*: \\"Microsoft in Talks to Acquire Gaming Company for $2 Billion\\"  \\n\*\*Output\*\*: \\"Business\\"\\n\\n\*\*Input\*\*: \\"Manchester United Secures Win in Premier League Football Match\\"  \\n\*\*Output\*\*: \\"Sports\\" \\n\\n# Notes\\n\\n- If the headline appears to fit into more than one category, choose the most dominant theme.\\n- Keywords or phrases such as \\"stocks\\", \\"company acquisition\\", \\"match\\", or technological brands can be good indicators for classification.\\n",

&nbsp;           "tool\_call\_id": null,

&nbsp;           "tool\_calls": null,

&nbsp;           "function\_call": null

&nbsp;         },

&nbsp;         {

&nbsp;           "role": "user",

&nbsp;           "content": "Stock Markets Rally After Positive Economic Data Released",

&nbsp;           "tool\_call\_id": null,

&nbsp;           "tool\_calls": null,

&nbsp;           "function\_call": null

&nbsp;         }

&nbsp;       ],

&nbsp;       "output": \[

&nbsp;         {

&nbsp;           "role": "assistant",

&nbsp;           "content": "Markets",

&nbsp;           "tool\_call\_id": null,

&nbsp;           "tool\_calls": null,

&nbsp;           "function\_call": null

&nbsp;         }

&nbsp;       ],

&nbsp;       "finish\_reason": "stop",

&nbsp;       "model": "gpt-4o-mini-2024-07-18",

&nbsp;       "usage": {

&nbsp;         "total\_tokens": 325,

&nbsp;         "completion\_tokens": 2,

&nbsp;         "prompt\_tokens": 323,

&nbsp;         "cached\_tokens": 0

&nbsp;       },

&nbsp;       "error": null,

&nbsp;       "temperature": 1.0,

&nbsp;       "max\_completion\_tokens": 2048,

&nbsp;       "top\_p": 1.0,

&nbsp;       "seed": 42

&nbsp;     }

&nbsp;   }

&nbsp; ],

&nbsp; "first\_id": "outputitem\_67e5796c28e081909917bf79f6e6214d",

&nbsp; "last\_id": "outputitem\_67e5796c28e081909917bf79f6e6214d",

&nbsp; "has\_more": true

}

The eval object

An Eval object with a data source config and testing criteria. An Eval represents a task to be done for your LLM integration. Like:



Improve the quality of my chatbot

See how well my chatbot handles customer support

Check if o4-mini is better at my usecase than gpt-4o

created\_at

integer



The Unix timestamp (in seconds) for when the eval was created.



data\_source\_config

object



Configuration of data sources used in runs of the evaluation.





Show possible types

id

string



Unique identifier for the evaluation.



metadata

map



Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard.



Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.



name

string



The name of the evaluation.



object

string



The object type.



testing\_criteria

array



A list of testing criteria.





Show possible types

OBJECT The eval object

{

&nbsp; "object": "eval",

&nbsp; "id": "eval\_67abd54d9b0081909a86353f6fb9317a",

&nbsp; "data\_source\_config": {

&nbsp;   "type": "custom",

&nbsp;   "item\_schema": {

&nbsp;     "type": "object",

&nbsp;     "properties": {

&nbsp;       "label": {"type": "string"},

&nbsp;     },

&nbsp;     "required": \["label"]

&nbsp;   },

&nbsp;   "include\_sample\_schema": true

&nbsp; },

&nbsp; "testing\_criteria": \[

&nbsp;   {

&nbsp;     "name": "My string check grader",

&nbsp;     "type": "string\_check",

&nbsp;     "input": "{{sample.output\_text}}",

&nbsp;     "reference": "{{item.label}}",

&nbsp;     "operation": "eq",

&nbsp;   }

&nbsp; ],

&nbsp; "name": "External Data Eval",

&nbsp; "created\_at": 1739314509,

&nbsp; "metadata": {

&nbsp;   "test": "synthetics",

&nbsp; }

}

The eval run object

A schema representing an evaluation run.



created\_at

integer



Unix timestamp (in seconds) when the evaluation run was created.



data\_source

object



Information about the run's data source.





Show possible types

error

object



An object representing an error response from the Eval API.





Show properties

eval\_id

string



The identifier of the associated evaluation.



id

string



Unique identifier for the evaluation run.



metadata

map



Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard.



Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.



model

string



The model that is evaluated, if applicable.



name

string



The name of the evaluation run.



object

string



The type of the object. Always "eval.run".



per\_model\_usage

array



Usage statistics for each model during the evaluation run.





Show properties

per\_testing\_criteria\_results

array



Results per testing criteria applied during the evaluation run.





Show properties

report\_url

string



The URL to the rendered evaluation run report on the UI dashboard.



result\_counts

object



Counters summarizing the outcomes of the evaluation run.





Show properties

status

string



The status of the evaluation run.



OBJECT The eval run object

{

&nbsp; "object": "eval.run",

&nbsp; "id": "evalrun\_67e57965b480819094274e3a32235e4c",

&nbsp; "eval\_id": "eval\_67e579652b548190aaa83ada4b125f47",

&nbsp; "report\_url": "https://platform.openai.com/evaluations/eval\_67e579652b548190aaa83ada4b125f47?run\_id=evalrun\_67e57965b480819094274e3a32235e4c",

&nbsp; "status": "queued",

&nbsp; "model": "gpt-4o-mini",

&nbsp; "name": "gpt-4o-mini",

&nbsp; "created\_at": 1743092069,

&nbsp; "result\_counts": {

&nbsp;   "total": 0,

&nbsp;   "errored": 0,

&nbsp;   "failed": 0,

&nbsp;   "passed": 0

&nbsp; },

&nbsp; "per\_model\_usage": null,

&nbsp; "per\_testing\_criteria\_results": null,

&nbsp; "data\_source": {

&nbsp;   "type": "completions",

&nbsp;   "source": {

&nbsp;     "type": "file\_content",

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Tech Company Launches Advanced Artificial Intelligence Platform",

&nbsp;           "ground\_truth": "Technology"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Central Bank Increases Interest Rates Amid Inflation Concerns",

&nbsp;           "ground\_truth": "Markets"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "International Summit Addresses Climate Change Strategies",

&nbsp;           "ground\_truth": "World"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Major Retailer Reports Record-Breaking Holiday Sales",

&nbsp;           "ground\_truth": "Business"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "National Team Qualifies for World Championship Finals",

&nbsp;           "ground\_truth": "Sports"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Stock Markets Rally After Positive Economic Data Released",

&nbsp;           "ground\_truth": "Markets"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Global Manufacturer Announces Merger with Competitor",

&nbsp;           "ground\_truth": "Business"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Breakthrough in Renewable Energy Technology Unveiled",

&nbsp;           "ground\_truth": "Technology"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "World Leaders Sign Historic Climate Agreement",

&nbsp;           "ground\_truth": "World"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Professional Athlete Sets New Record in Championship Event",

&nbsp;           "ground\_truth": "Sports"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Financial Institutions Adapt to New Regulatory Requirements",

&nbsp;           "ground\_truth": "Business"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Tech Conference Showcases Advances in Artificial Intelligence",

&nbsp;           "ground\_truth": "Technology"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Global Markets Respond to Oil Price Fluctuations",

&nbsp;           "ground\_truth": "Markets"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "International Cooperation Strengthened Through New Treaty",

&nbsp;           "ground\_truth": "World"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "item": {

&nbsp;           "input": "Sports League Announces Revised Schedule for Upcoming Season",

&nbsp;           "ground\_truth": "Sports"

&nbsp;         }

&nbsp;       }

&nbsp;     ]

&nbsp;   },

&nbsp;   "input\_messages": {

&nbsp;     "type": "template",

&nbsp;     "template": \[

&nbsp;       {

&nbsp;         "type": "message",

&nbsp;         "role": "developer",

&nbsp;         "content": {

&nbsp;           "type": "input\_text",

&nbsp;           "text": "Categorize a given news headline into one of the following topics: Technology, Markets, World, Business, or Sports.\\n\\n# Steps\\n\\n1. Analyze the content of the news headline to understand its primary focus.\\n2. Extract the subject matter, identifying any key indicators or keywords.\\n3. Use the identified indicators to determine the most suitable category out of the five options: Technology, Markets, World, Business, or Sports.\\n4. Ensure only one category is selected per headline.\\n\\n# Output Format\\n\\nRespond with the chosen category as a single word. For instance: \\"Technology\\", \\"Markets\\", \\"World\\", \\"Business\\", or \\"Sports\\".\\n\\n# Examples\\n\\n\*\*Input\*\*: \\"Apple Unveils New iPhone Model, Featuring Advanced AI Features\\"  \\n\*\*Output\*\*: \\"Technology\\"\\n\\n\*\*Input\*\*: \\"Global Stocks Mixed as Investors Await Central Bank Decisions\\"  \\n\*\*Output\*\*: \\"Markets\\"\\n\\n\*\*Input\*\*: \\"War in Ukraine: Latest Updates on Negotiation Status\\"  \\n\*\*Output\*\*: \\"World\\"\\n\\n\*\*Input\*\*: \\"Microsoft in Talks to Acquire Gaming Company for $2 Billion\\"  \\n\*\*Output\*\*: \\"Business\\"\\n\\n\*\*Input\*\*: \\"Manchester United Secures Win in Premier League Football Match\\"  \\n\*\*Output\*\*: \\"Sports\\" \\n\\n# Notes\\n\\n- If the headline appears to fit into more than one category, choose the most dominant theme.\\n- Keywords or phrases such as \\"stocks\\", \\"company acquisition\\", \\"match\\", or technological brands can be good indicators for classification.\\n"

&nbsp;         }

&nbsp;       },

&nbsp;       {

&nbsp;         "type": "message",

&nbsp;         "role": "user",

&nbsp;         "content": {

&nbsp;           "type": "input\_text",

&nbsp;           "text": "{{item.input}}"

&nbsp;         }

&nbsp;       }

&nbsp;     ]

&nbsp;   },

&nbsp;   "model": "gpt-4o-mini",

&nbsp;   "sampling\_params": {

&nbsp;     "seed": 42,

&nbsp;     "temperature": 1.0,

&nbsp;     "top\_p": 1.0,

&nbsp;     "max\_completions\_tokens": 2048

&nbsp;   }

&nbsp; },

&nbsp; "error": null,

&nbsp; "metadata": {}

}

The eval run output item object

A schema representing an evaluation run output item.



created\_at

integer



Unix timestamp (in seconds) when the evaluation run was created.



datasource\_item

object



Details of the input data source item.



datasource\_item\_id

integer



The identifier for the data source item.



eval\_id

string



The identifier of the evaluation group.



id

string



Unique identifier for the evaluation run output item.



object

string



The type of the object. Always "eval.run.output\_item".



results

array



A list of grader results for this output item.





Show properties

run\_id

string



The identifier of the evaluation run associated with this output item.



sample

object



A sample containing the input and output of the evaluation run.





Show properties

status

string



The status of the evaluation run.



OBJECT The eval run output item object

{

&nbsp; "object": "eval.run.output\_item",

&nbsp; "id": "outputitem\_67abd55eb6548190bb580745d5644a33",

&nbsp; "run\_id": "evalrun\_67abd54d60ec8190832b46859da808f7",

&nbsp; "eval\_id": "eval\_67abd54d9b0081909a86353f6fb9317a",

&nbsp; "created\_at": 1739314509,

&nbsp; "status": "pass",

&nbsp; "datasource\_item\_id": 137,

&nbsp; "datasource\_item": {

&nbsp;     "teacher": "To grade essays, I only check for style, content, and grammar.",

&nbsp;     "student": "I am a student who is trying to write the best essay."

&nbsp; },

&nbsp; "results": \[

&nbsp;   {

&nbsp;     "name": "String Check Grader",

&nbsp;     "type": "string-check-grader",

&nbsp;     "score": 1.0,

&nbsp;     "passed": true,

&nbsp;   }

&nbsp; ],

&nbsp; "sample": {

&nbsp;   "input": \[

&nbsp;     {

&nbsp;       "role": "system",

&nbsp;       "content": "You are an evaluator bot..."

&nbsp;     },

&nbsp;     {

&nbsp;       "role": "user",

&nbsp;       "content": "You are assessing..."

&nbsp;     }

&nbsp;   ],

&nbsp;   "output": \[

&nbsp;     {

&nbsp;       "role": "assistant",

&nbsp;       "content": "The rubric is not clear nor concise."

&nbsp;     }

&nbsp;   ],

&nbsp;   "finish\_reason": "stop",

&nbsp;   "model": "gpt-4o-2024-08-06",

&nbsp;   "usage": {

&nbsp;     "total\_tokens": 521,

&nbsp;     "completion\_tokens": 2,

&nbsp;     "prompt\_tokens": 519,

&nbsp;     "cached\_tokens": 0

&nbsp;   },

&nbsp;   "error": null,

&nbsp;   "temperature": 1.0,

&nbsp;   "max\_completion\_tokens": 2048,

&nbsp;   "top\_p": 1.0,

&nbsp;   "seed": 42

&nbsp; }

}

Previous

Embeddings

Next

Fine-tuning

