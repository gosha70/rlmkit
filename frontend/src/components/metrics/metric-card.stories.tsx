import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { Hash, DollarSign, Clock, TrendingUp } from "lucide-react";
import { MetricCard } from "./metric-card";

const meta: Meta<typeof MetricCard> = {
  title: "Metrics/MetricCard",
  component: MetricCard,
};

export default meta;
type Story = StoryObj<typeof MetricCard>;

export const Tokens: Story = { args: { label: "Total Tokens", value: "12,450", icon: Hash } };
export const Cost: Story = { args: { label: "Total Cost", value: "$0.0342", icon: DollarSign } };
export const Latency: Story = { args: { label: "Avg Latency", value: "2.3", unit: "s", icon: Clock } };
export const Savings: Story = { args: { label: "Token Savings", value: "34%", icon: TrendingUp } };
