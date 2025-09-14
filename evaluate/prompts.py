SYSTEM_INSTRUCTION = """I will provide you with the original code before editing, \
its recent edits, and its current version. Your goals are: consider what edits \
need to be made next, make sure the next changes are relevant, keep the code \
editing intent the same.

Here is a simple example:
<original_code>
import re

def split_string(string, seperator=' '):
    return string.split(seperator) 

def split_regex(string, seperator_pattern):
    return re.split(seperator_pattern, string)

class FilterModule(object):
    ''' A filter to split a string into a list. '''
    def filters(self):
        return {
            'split' : split_string,
            'split_regex' : split_regex,
        }
</original_code>
<edits_diff>
@@ -1,0 +1,1 @@
+ from ansible import errors
@@ -4,1 +5,4 @@
+     try:
-     return string.split(seperator)
+         return string.split(seperator)
+     except Exception, e:
+         raise errors.AnsibleFilterError('split plugin error: %s, string=%s' % str(e),str(string) )
</edits_diff>
<current_version>
from ansible import errors
import re

def split_string(string, seperator=' '):
    try:
        return string.split(seperator)
    except Exception, e:
        raise errors.AnsibleFilterError('split plugin error: %s, string=%s' % str(e),str(string) )

def split_regex(string, seperator_pattern):
    return re.split(seperator_pattern, string)

class FilterModule(object):
    ''' A filter to split a string into a list. '''
    def filters(self):
        return {
            'split' : split_string,
            'split_regex' : split_regex,
        }
</current_version>
<next_version>
from ansible import errors
import re

def split_string(string, seperator=' '):
    try:
        return string.split(seperator)
    except Exception, e:
        raise errors.AnsibleFilterError('split plugin error: %s, string=%s' % str(e),str(string) )

def split_regex(string, seperator_pattern):
    try:
        return re.split(seperator_pattern, string)
    except Exception as e:
        raise errors.AnsibleFilterError('split plugin error: %s, string=%s' % str(e),str(string) )

class FilterModule(object):
    ''' A filter to split a string into a list. '''
    def filters(self):
        return {
            'split' : split_string,
            'split_regex' : split_regex,
        }
</next_version>
"""

PROMPT_TEMPLATE = """This is the original code before editing:
<original_code>
{original_code}
</original_code>
This is a sequence of edits that I made on it:
<edits_diff>
{edits}
</edits_diff>
This is what the piece of code looks like now:
<current_version>
{current_version}
</current_version>

Based on my most recent edits, what will I do next? Please rewrite the code \
between <current_version> and </current_version> based on what I will do next. \ 
Note that you MUST follow the following rules:
- Your response MUST begin with <next_version> and end with </next_version>.
- Only answer with the updated code. NO EXPLANATION of the generated code is required.
- Make sure that the indentation level of any new code is correct and \
consistent with the existing code. Do not skip any lines.
"""

LLM_AS_A_JUDGE_PROMPT = """### Task Description
I'm evaluating the model's ability to make code edit predictions. \
I will give you the original prompt, please compare the ground truth with the \
model's output to determine whether the model gives a reasonable prediction, \
and respond with "yes" or "no" accordingly, without any additional explanation.

### Original Prompt
{prompt}

### Ground Truth
{ground_truth}

### Model Output
{model_output}
"""