import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { Badge } from "./badge";

const meta: Meta<typeof Badge> = {
  title: "UI/Badge",
  component: Badge,
  args: { children: "Badge" },
};

export default meta;
type Story = StoryObj<typeof Badge>;

export const Default: Story = {};
export const Secondary: Story = { args: { variant: "secondary" } };
export const Destructive: Story = { args: { variant: "destructive" } };
export const Outline: Story = { args: { variant: "outline" } };
export const Success: Story = { args: { variant: "success", children: "Connected" } };
export const Warning: Story = { args: { variant: "warning", children: "Slow" } };
