import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { StatusIndicator } from "./status-indicator";

const meta: Meta<typeof StatusIndicator> = {
  title: "Shared/StatusIndicator",
  component: StatusIndicator,
};

export default meta;
type Story = StoryObj<typeof StatusIndicator>;

export const Connected: Story = { args: { status: "connected" } };
export const Offline: Story = { args: { status: "offline" } };
export const Warning: Story = { args: { status: "warning" } };
export const Loading: Story = { args: { status: "loading" } };
export const Configured: Story = { args: { status: "configured" } };
export const NotConfigured: Story = { args: { status: "not_configured" } };
export const WithLabel: Story = { args: { status: "connected", label: "OpenAI" } };
