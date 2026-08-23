
<img width="966" height="665" alt="image" src="https://github.com/user-attachments/assets/c2331103-0365-4337-ab4f-bff7dea1a917" />
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
