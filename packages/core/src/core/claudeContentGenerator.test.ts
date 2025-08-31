import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ClaudeContentGenerator } from './claudeContentGenerator.js';
import { GenerateContentParameters, FinishReason } from '@google/genai';
import { AuthType } from './contentGenerator.js';

// Mock the ClaudeAdapter
vi.mock('../adapters/claudeAdapter.js', () => ({
  ClaudeAdapter: vi.fn().mockImplementation(() => ({
    generateContent: vi.fn(),
    generateContentStream: vi.fn(),
    countTokens: vi.fn(),
    embedContent: vi.fn(),
  })),
}));

describe('ClaudeContentGenerator', () => {
  let generator: ClaudeContentGenerator;
  let mockAdapter: any;

  beforeEach(() => {
    // Set up environment variables
    process.env['ANTHROPIC_VERTEX_PROJECT_ID'] = 'test-project';
    process.env['CLOUD_ML_REGION'] = 'us-central1';

    // Create generator instance
    const config = {
      model: 'claude-3-5-sonnet-v2@20241022',
      authType: AuthType.USE_CLAUDE,
    };

    generator = new ClaudeContentGenerator(config);

    // Get the mocked adapter instance
    mockAdapter = (generator as any).adapter;
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('generateContent', () => {
    it('should delegate to ClaudeAdapter', async () => {
      const mockResponse = {
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
      };

      mockAdapter.generateContent.mockResolvedValue(mockResponse);

      const request: GenerateContentParameters = {
        model: 'claude-3-5-sonnet-v2@20241022',
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Hello Claude!' }],
          },
        ],
      };

      const result = await generator.generateContent(request, 'test-prompt-id');

      expect(mockAdapter.generateContent).toHaveBeenCalledWith(request);
      expect(result).toEqual(mockResponse);
    });
  });

  describe('generateContentStream', () => {
    it('should delegate to ClaudeAdapter for streaming', async () => {
      const mockStreamResponse = [
        {
          candidates: [
            {
              content: {
                parts: [{ text: 'Hello' }],
                role: 'model',
              },
              finishReason: null,
              index: 0,
              safetyRatings: [],
            },
          ],
          promptFeedback: {
            safetyRatings: [],
          },
        },
        {
          candidates: [
            {
              content: {
                parts: [{ text: ' from Claude!' }],
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
        },
      ];

      async function* mockStreamGenerator() {
        for (const response of mockStreamResponse) {
          yield response;
        }
      }

      mockAdapter.generateContentStream.mockResolvedValue(mockStreamGenerator());

      const request: GenerateContentParameters = {
        model: 'claude-3-5-sonnet-v2@20241022',
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Hello Claude!' }],
          },
        ],
      };

      const streamGenerator = await generator.generateContentStream(request, 'test-prompt-id');
      const results = [];
      
      for await (const chunk of streamGenerator) {
        results.push(chunk);
      }

      expect(mockAdapter.generateContentStream).toHaveBeenCalledWith(request);
      expect(results).toEqual(mockStreamResponse);
    });
  });

  describe('countTokens', () => {
    it('should delegate to ClaudeAdapter', async () => {
      const mockTokenResponse = {
        totalTokens: 42,
      };

      mockAdapter.countTokens.mockResolvedValue(mockTokenResponse);

      const request = {
        model: 'claude-3-5-sonnet-v2@20241022',
        contents: [
          {
            role: 'user' as const,
            parts: [{ text: 'Test message' }],
          },
        ],
      };

      const result = await generator.countTokens(request);

      expect(mockAdapter.countTokens).toHaveBeenCalledWith(request);
      expect(result).toEqual(mockTokenResponse);
    });
  });

  describe('embedContent', () => {
    it('should delegate to ClaudeAdapter', async () => {
      const mockEmbedResponse = {
        embedding: {
          values: [0.1, 0.2, 0.3],
        },
      };

      mockAdapter.embedContent.mockResolvedValue(mockEmbedResponse);

      const request = {
        model: 'text-embedding-ada-002',
        contents: ['test text'],
      };

      const result = await generator.embedContent(request);

      expect(mockAdapter.embedContent).toHaveBeenCalledWith(request);
      expect(result).toEqual(mockEmbedResponse);
    });

    it('should handle embedding errors gracefully', async () => {
      mockAdapter.embedContent.mockRejectedValue(new Error('Embedding not supported'));

      const request = {
        model: 'text-embedding-ada-002',
        contents: ['test text'],
      };

      await expect(generator.embedContent(request)).rejects.toThrow('Embedding not supported');
    });
  });

  describe('environment validation', () => {
    it('should validate required environment variables', () => {
      const originalEnv = process.env;
      
      // Clear environment variables
      delete process.env['ANTHROPIC_VERTEX_PROJECT_ID'];
      
      expect(() => {
        new ClaudeContentGenerator({
          model: 'claude-3-5-sonnet-v2@20241022',
          authType: AuthType.USE_CLAUDE,
        });
      }).toThrow('ANTHROPIC_VERTEX_PROJECT_ID environment variable is required');

      // Restore environment
      process.env = originalEnv;
    });
  });
});
