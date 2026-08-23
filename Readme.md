
<img width="966" height="665" alt="image" src="https://github.com/user-attachments/assets/c2331103-0365-4337-ab4f-bff7dea1a917" />
                    ┌──────────────────────┐
                    │      Voice Input     │
                    └──────────┬───────────┘
                               ↓
                         sarvamAI
                               ↓
                          Transcript
                               ↓
                    ┌──────────────────────┐
                    │      LangGraph       │
                    │   Agent Orchestrator │
                    └──────────┬───────────┘
                               ↓
                          ┌─────────┐
                          │ gemini   │
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
                           gemini
                              ↓
                        Final Response
                              ↓
                            TTS
                              ↓
                       Streaming Audio
                              ↓
                            User
