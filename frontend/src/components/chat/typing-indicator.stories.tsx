import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { TypingIndicator } from "./typing-indicator";

const meta: Meta<typeof TypingIndicator> = {
  title: "Chat/TypingIndicator",
  component: TypingIndicator,
};

export default meta;
type Story = StoryObj<typeof TypingIndicator>;

export const Default: Story = {};
export const WithMode: Story = { args: { mode: "rlm" } };
export const WithStep: Story = { args: { step: "Running code..." } };
