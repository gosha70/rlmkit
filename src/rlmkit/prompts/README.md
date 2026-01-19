# RLM Prompt Templates

This directory contains versioned, discoverable prompt templates for the RLM system.

## Naming Convention

Template files follow a strict naming convention for discoverability:

```
{prompt_type}_v{major}_{minor}.{ext}
```

**Examples:**
- `system_prompt_v1_0.txt` - System prompt version 1.0
- `system_prompt_v2_0.txt` - System prompt version 2.0
- `system_prompt_v1_1.txt` - System prompt version 1.1

## File Formats

Supported formats:
- `.txt` - Plain text templates (recommended)
- `.md` - Markdown templates
- `.yaml` / `.yml` - YAML files with a `template` key

## Template Variables

Templates use Python's `.format()` syntax for variable substitution:

**Available variables:**
- `{prompt_length}` - Length of the content being analyzed (formatted with commas)
- Custom variables can be passed via `**kwargs`

**Example template:**
```
You are analyzing {prompt_length:,} characters of text.
```

## Usage

### Load default versioned template:
```python
from rlmkit import get_default_system_prompt, format_system_prompt

# Get template v1.0
template = get_default_system_prompt("1.0")

# Format with variables
prompt = format_system_prompt(prompt_length=12345)
```

### Use custom template:
```python
custom_template = "Analyze {prompt_length} chars..."
prompt = format_system_prompt(
    template=custom_template,
    prompt_length=1000
)
```

### Load from custom file:
```python
from pathlib import Path
from rlmkit.prompts.templates import load_prompt_from_file

template = load_prompt_from_file(Path("my_custom_prompt.txt"))
prompt = format_system_prompt(template=template, prompt_length=5000)
```

## Creating New Versions

To create a new version:

1. **Copy existing template:**
   ```bash
   cp system_prompt_v1_0.txt system_prompt_v2_0.txt
   ```

2. **Edit the new version** with your changes

3. **Use the new version:**
   ```python
   prompt = format_system_prompt(version="2.0", prompt_length=1000)
   ```

## Version History

### v1.0 (Current)
- Initial RLM system prompt
- Supports: peek, grep, select, chunk tools
- Uses FINAL and FINAL_VAR directives

## Model-Specific Prompts

For model-specific optimizations, create separate template files:

```
system_prompt_v1_0_gpt4.txt
system_prompt_v1_0_claude.txt
system_prompt_v1_0_llama.txt
```

Load with:
```python
# Custom loading logic
template = load_prompt_from_file(Path(f"system_prompt_v1_0_{model_name}.txt"))
```

## Best Practices

1. **Always version your prompts** - Use semantic versioning (major.minor)
2. **Keep templates in version control** - Track changes over time
3. **Document changes** - Update this README when adding new versions
4. **Test new versions** - Run test suite after prompt changes
5. **Use meaningful names** - Follow the naming convention strictly
