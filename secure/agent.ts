/**
 * Moltbot Secure - Agent Core
 *
 * Minimal AI agent that handles conversations.
 * Direct API calls to Anthropic or OpenAI - no intermediaries.
 */

import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
import type { SecureConfig } from "./config.js";
import type { AuditLogger } from "./audit.js";

export type Message = {
  role: "user" | "assistant";
  content: string;
};

export type AgentResponse = {
  text: string;
  usage?: {
    inputTokens: number;
    outputTokens: number;
  };
};

export type AgentCore = {
  chat: (messages: Message[], systemPrompt?: string) => Promise<AgentResponse>;
  provider: "anthropic" | "openai";
};

const DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514";
const DEFAULT_OPENAI_MODEL = "gpt-4o";

const DEFAULT_SYSTEM_PROMPT = `You are a helpful AI assistant running as a secure, self-hosted bot.

You are direct, concise, and helpful. You can:
- Answer questions and have conversations
- Analyze images and documents shared with you
- Help with coding and technical tasks
- Summarize content and extract information

When you receive webhook notifications, summarize them helpfully for the user.

Be security-conscious:
- Never reveal API keys, tokens, or secrets
- Don't execute commands that could harm the system
- Warn users about potentially dangerous operations`;

function createAnthropicAgent(config: SecureConfig, audit: AuditLogger): AgentCore {
  const client = new Anthropic({
    apiKey: config.ai.apiKey,
  });

  const model = config.ai.model || DEFAULT_ANTHROPIC_MODEL;

  return {
    provider: "anthropic",
    async chat(messages: Message[], systemPrompt?: string): Promise<AgentResponse> {
      try {
        const response = await client.messages.create({
          model,
          max_tokens: 4096,
          system: systemPrompt || DEFAULT_SYSTEM_PROMPT,
          messages: messages.map((m) => ({
            role: m.role,
            content: m.content,
          })),
        });

        const text = response.content
          .filter((block): block is Anthropic.TextBlock => block.type === "text")
          .map((block) => block.text)
          .join("\n");

        return {
          text,
          usage: {
            inputTokens: response.usage.input_tokens,
            outputTokens: response.usage.output_tokens,
          },
        };
      } catch (err) {
        audit.error({
          error: `Anthropic API error: ${err instanceof Error ? err.message : String(err)}`,
        });
        throw err;
      }
    },
  };
}

function createOpenAIAgent(config: SecureConfig, audit: AuditLogger): AgentCore {
  const client = new OpenAI({
    apiKey: config.ai.apiKey,
  });

  const model = config.ai.model || DEFAULT_OPENAI_MODEL;

  return {
    provider: "openai",
    async chat(messages: Message[], systemPrompt?: string): Promise<AgentResponse> {
      try {
        const response = await client.chat.completions.create({
          model,
          max_tokens: 4096,
          messages: [
            { role: "system", content: systemPrompt || DEFAULT_SYSTEM_PROMPT },
            ...messages.map((m) => ({
              role: m.role as "user" | "assistant",
              content: m.content,
            })),
          ],
        });

        const text = response.choices[0]?.message?.content || "";

        return {
          text,
          usage: response.usage
            ? {
                inputTokens: response.usage.prompt_tokens,
                outputTokens: response.usage.completion_tokens,
              }
            : undefined,
        };
      } catch (err) {
        audit.error({
          error: `OpenAI API error: ${err instanceof Error ? err.message : String(err)}`,
        });
        throw err;
      }
    },
  };
}

export function createAgent(config: SecureConfig, audit: AuditLogger): AgentCore {
  if (config.ai.provider === "anthropic") {
    return createAnthropicAgent(config, audit);
  }
  return createOpenAIAgent(config, audit);
}

/**
 * Simple in-memory conversation store
 * For Railway, consider using Redis or persistent storage
 */
export type ConversationStore = {
  get: (userId: number) => Message[];
  add: (userId: number, message: Message) => void;
  clear: (userId: number) => void;
};

const MAX_HISTORY = 20;

export function createConversationStore(): ConversationStore {
  const conversations = new Map<number, Message[]>();

  return {
    get(userId: number): Message[] {
      return conversations.get(userId) || [];
    },

    add(userId: number, message: Message): void {
      const history = conversations.get(userId) || [];
      history.push(message);
      // Keep only last N messages
      if (history.length > MAX_HISTORY) {
        history.splice(0, history.length - MAX_HISTORY);
      }
      conversations.set(userId, history);
    },

    clear(userId: number): void {
      conversations.delete(userId);
    },
  };
}
