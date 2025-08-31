import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ClaudeAdapter } from './claudeAdapter.js';
import { GenerateContentParameters, FinishReason, Type } from '@google/genai';

describe('ClaudeAdapter', () => {
  let adapter: ClaudeAdapter;

  beforeEach(() => {
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
            text: 'Hello! How can I help you today?',
          },
        ],
        stop_reason: 'end_turn',
      };

      // Mock the Claude client
      vi.spyOn(adapter['client'].messages, 'create').mockResolvedValue(mockClaudeResponse as any);

      const request: GenerateContentParameters = {
        model: 'claude-3-7-sonnet',
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Hello' }],
          },
        ],
      };

      const result = await adapter.generateContent(request);

      expect(result.candidates).toBeDefined();
      expect(result.candidates?.[0]?.content?.parts?.[0]).toEqual({
        text: 'Hello! How can I help you today?',
      });
      expect(result.candidates![0].finishReason).toBe(FinishReason.STOP);
    });

    it('should handle system instructions', async () => {
      const mockClaudeResponse = {
        content: [{ type: 'text', text: 'Response with system context' }],
        stop_reason: 'end_turn',
      };

      const createSpy = vi.spyOn(adapter['client'].messages, 'create').mockResolvedValue(mockClaudeResponse as any);

      const request: GenerateContentParameters = {
        model: 'claude-3-7-sonnet',
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Hello' }],
          },
        ],
        config: {
          systemInstruction: {
            parts: [{ text: 'You are a helpful assistant.' }],
          },
        },
      };

      await adapter.generateContent(request);

      expect(createSpy).toHaveBeenCalledWith(
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
            name: 'get_weather',
            input: { location: 'San Francisco' },
          },
        ],
        stop_reason: 'tool_use',
      };

      vi.spyOn(adapter['client'].messages, 'create').mockResolvedValue(mockClaudeResponse as any);

      const request: GenerateContentParameters = {
        model: 'claude-3-7-sonnet',
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
                    type: Type.OBJECT,
                    properties: {
                      location: { type: Type.STRING },
                    },
                  },
                },
              ],
            },
          ],
        },
      };

      const result = await adapter.generateContent(request);

      expect(result.candidates).toBeDefined();
      expect(result.candidates?.[0]?.content?.parts?.[0]).toEqual({
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
        model: 'claude-3-7-sonnet',
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
        contents: 'test text',
      };

      await expect(adapter.embedContent(request)).rejects.toThrow(
        'Embedding not supported for Claude models'
      );
    });
  });
});
