import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { MessageBubble } from "./message-bubble";

const meta: Meta<typeof MessageBubble> = {
  title: "Chat/MessageBubble",
  component: MessageBubble,
};

export default meta;
type Story = StoryObj<typeof MessageBubble>;

export const UserMessage: Story = {
  args: {
    message: {
      id: "1",
      role: "user",
      content: "Explain how recursive language models work.",
    },
  },
};

export const AssistantMessage: Story = {
  args: {
    message: {
      id: "2",
      role: "assistant",
      content: "Recursive Language Models (RLMs) explore content through **iterative code generation**.\n\nEach step:\n1. Generates Python code to inspect the document\n2. Executes code in a sandbox\n3. Uses output to inform the next step",
      mode_used: "rlm",
      metrics: {
        total_tokens: 1250,
        input_tokens: 800,
        output_tokens: 450,
        cost_usd: 0.0034,
        elapsed_seconds: 3.2,
        steps: 4,
      },
    },
  },
};

export const Streaming: Story = {
  args: {
    message: {
      id: "3",
      role: "assistant",
      content: "",
      isStreaming: true,
    },
  },
};

export const MarkdownContent: Story = {
  args: {
    message: {
      id: "4",
      role: "assistant",
      content: "Here is a code example:\n\n```python\ndef hello():\n    print('world')\n```\n\n| Column A | Column B |\n|----------|----------|\n| Value 1  | Value 2  |",
      mode_used: "direct",
    },
  },
};
