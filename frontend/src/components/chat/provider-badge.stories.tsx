import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { ProviderBadge } from "./provider-badge";

const meta: Meta<typeof ProviderBadge> = {
  title: "Chat/ProviderBadge",
  component: ProviderBadge,
  args: { provider: "openai" },
};

export default meta;
type Story = StoryObj<typeof ProviderBadge>;

export const ProviderOnly: Story = {};
export const WithModel: Story = { args: { provider: "openai", model: "gpt-4o" } };
export const Offline: Story = { args: { provider: "anthropic", model: "claude-sonnet-4-5-20250514", status: "offline" } };
export const NotConfigured: Story = { args: { provider: "ollama", status: "not_configured" } };
