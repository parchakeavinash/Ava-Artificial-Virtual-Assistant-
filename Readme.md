                    ┌──────────────────────┐
                    │      Voice Input     │
                    └──────────┬───────────┘
                               ↓
                         AssemblyAI
                               ↓
                          Transcript
                               ↓
                    ┌──────────────────────┐
                    │      LangGraph       │
                    │   Agent Orchestrator │
                    └──────────┬───────────┘
                               ↓
                          ┌─────────┐
                          │ Qwen3   │
                          └────┬────┘
                               ↓
                       Tool decision
                               ↓
                       ┌─────────────┐
                       │ MCP Client  │
                       └──────┬──────┘
                              ↓
             ┌────────────────┼────────────────┐
             ↓                ↓                ↓
       Web MCP            DB MCP           Memory MCP
             ↓                ↓                ↓
        Firecrawl         Supabase          pgvector
             │                │                │
             └────────────────┼────────────────┘
                              ↓
                         Tool Results
                              ↓
                           Qwen3
                              ↓
                        Final Response
                              ↓
                            TTS
                              ↓
                       Streaming Audio
                              ↓
                            User