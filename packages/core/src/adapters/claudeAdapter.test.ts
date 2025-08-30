/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ClaudeAdapter } from './claudeAdapter.js';
import { GenerateContentParameters, FinishReason } from '@google/genai';

// Mock the AnthropicVertex SDK
vi.mock('@anthropic-ai/vertex-sdk', () => ({
  AnthropicVertex: vi.fn().mockImplementation(() => ({
    messages: {
      create: vi.fn(),
    },
  })),
}));

describe('ClaudeAdapter', () => {
  let adapter: ClaudeAdapter;
  let mockClient: any;

  beforeEach(() => {
    const { AnthropicVertex } = require('@anthropic-ai/vertex-sdk');
    mockClient = {
      messages: {
        create: vi.fn(),
      },
    };
    AnthropicVertex.mockReturnValue(mockClient);
    
    adapter = new ClaudeAdapter({
      region: 'us-central1',
      projectId: 'test-project',
    });
  });

  describe('generateContent', () => {
    it('should convert Gemini request to Claude format and back', async () => {
      const mockClaudeResponse = {
        content: [
          {
            type: 'text',
            text: 'Hello from Claude!',
          },
        ],
        stop_reason: 'end_turn',
      };

      mockClient.messages.create.mockResolvedValue(mockClaudeResponse);

      const request: GenerateContentParameters = {
        model: 'claude-3-5-sonnet-v2@20241022',
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Hello Claude!' }],
          },
        ],
        config: {
          temperature: 0.7,
          maxOutputTokens: 1000,
        },
      };

      const result = await adapter.generateContent(request);

      expect(mockClient.messages.create).toHaveBeenCalledWith({
        model: 'claude-3-5-sonnet-v2@20241022',
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: 'Hello Claude!',
              },
            ],
          },
        ],
        max_tokens: 1000,
        temperature: 0.7,
      });

      expect(result).toEqual({
        candidates: [
          {
            content: {
              parts: [{ text: 'Hello from Claude!' }],
              role: 'model',
            },
            finishReason: FinishReason.STOP,
            index: 0,
            safetyRatings: [],
          },
        ],
        promptFeedback: {
          safetyRatings: [],
        },
      });
    });

    it('should handle system instructions', async () => {
      const mockClaudeResponse = {
        content: [{ type: 'text', text: 'Response with system context' }],
        stop_reason: 'end_turn',
      };

      mockClient.messages.create.mockResolvedValue(mockClaudeResponse);

      const request: GenerateContentParameters = {
        model: 'claude-3-5-sonnet-v2@20241022',
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Hello' }],
          },
        ],
        config: {
          systemInstruction: 'You are a helpful assistant.',
        },
      };

      await adapter.generateContent(request);

      expect(mockClient.messages.create).toHaveBeenCalledWith(
        expect.objectContaining({
          system: 'You are a helpful assistant.',
        })
      );
    });

    it('should handle function calls', async () => {
      const mockClaudeResponse = {
        content: [
          {
            type: 'tool_use',
            id: 'tool_123',
            name: 'get_weather',
            input: { location: 'San Francisco' },
          },
        ],
        stop_reason: 'tool_use',
      };

      mockClient.messages.create.mockResolvedValue(mockClaudeResponse);

      const request: GenerateContentParameters = {
        model: 'claude-3-5-sonnet-v2@20241022',
        contents: [
          {
            role: 'user',
            parts: [{ text: 'What is the weather in San Francisco?' }],
          },
        ],
        config: {
          tools: [
            {
              functionDeclarations: [
                {
                  name: 'get_weather',
                  description: 'Get weather information',
                  parameters: {
                    type: 'object',
                    properties: {
                      location: { type: 'string' },
                    },
                  },
                },
              ],
            },
          ],
        },
      };

      const result = await adapter.generateContent(request);

      expect(result.candidates[0].content.parts[0]).toEqual({
        functionCall: {
          name: 'get_weather',
          args: { location: 'San Francisco' },
        },
      });
    });
  });

  describe('countTokens', () => {
    it('should provide approximate token count', async () => {
      const request = {
        contents: [
          {
            role: 'user' as const,
            parts: [{ text: 'This is a test message with some words.' }],
          },
        ],
      };

      const result = await adapter.countTokens(request);

      expect(result.totalTokens).toBeGreaterThan(0);
      expect(typeof result.totalTokens).toBe('number');
    });
  });

  describe('embedContent', () => {
    it('should throw error for unsupported embedding', async () => {
      const request = {
        model: 'text-embedding-ada-002',
        contents: ['test text'],
      };

      await expect(adapter.embedContent(request)).rejects.toThrow(
        'Embedding not supported for Claude models'
      );
    });
  });
});
