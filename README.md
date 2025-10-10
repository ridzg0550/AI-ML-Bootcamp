Neuromorphic Memory Architecture for Interactive Narrative Generation

A biologically-inspired memory system for long-horizon interactive storytelling in text-based RPGs. This architecture integrates neuromorphic episodic memory with transformer-based LLMs to maintain coherent narratives across extended conversations spanning multiple sessions.


🌟 Key Features

Neuromorphic Event Memory: Salience-based episodic storage with attention-driven retrieval

Associative Memory Networks: Spreading activation enables recall of narratively connected events

Multi-Factor Retrieval: Combines semantic similarity, temporal recency, salience, and permanence scoring

Memory Consolidation: Biologically-inspired link pruning and strengthening

Specialized Subsystems: Dedicated tracking for NPCs, quests, and world state

Session Persistence: Complete state save/load for multi-session campaigns

Deterministic Execution: Seeded randomness ensures reproducible behavior


🏗️ Architecture Overview

Player Input → Query Encoding → Memory Retrieval → Context Assembly → LLM Generation

 Memory Retrieval :-
            1. Neuromorphic Event Memory            2.Specialized Subsystems
            • Vector Embeddings (FAISS)             • NPC Memory
            • Salience Filtering                    • Quest Log
            • Associative Linking                   • Slot Memory
            • Consolidation                         • State Management

                                


🌟Core Components

1.Neuromorphic Event Memory

384-dimensional semantic embeddings (Sentence-BERT)

FAISS indexing for efficient retrieval

Salience-based filtering prevents memory bloat

Bidirectional associative links enable graph-based chaining


2.NPC Memory System

Structured character profiles with temporal metadata

Relationship state tracking (friendly, neutral, hostile, allied)

Named Entity Recognition for automatic character detection


3.Quest Log

Regex-based quest detection with validation

State management (active, completed, failed)

Prevents false positives through semantic filtering


4.Slot Memory

Lightweight environmental context (location, time, party, inventory)

Regex extraction for state changes


🌟Prerequisites

 bash
Python 3.8+
pip

🌟Required Dependencies

sentence-transformers==2.2.2
faiss-cpu==1.7.4
spacy==3.5.0
groq==0.4.0
numpy==1.24.3
python-dotenv==1.0.0

🌟How To Run
1.paste the colab code
2.paste the API key in the mentioned place
3.run
4.give inputs as a player and enjoy the game!

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
            
