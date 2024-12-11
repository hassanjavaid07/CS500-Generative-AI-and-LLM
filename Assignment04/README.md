This assignment is built on a Multi-Agentic Retrieival Augmented Generation (RAG) to perform document-related and internet search related queries. The assigned document was NetSol financial statement report. 

The Hugging Face deployment can be found [here](https://huggingface.co/spaces/trident-10/Researcher-RAG/tree/main).

The RAGAS metrics are uploaded in the file `rag-metric-dataframe.csv`. The metrics calculated were `faithfulness` and `answer_correctness`. However due to presistent `RateErrorLimit`, the metrics could only be calculated for a few entries.

The output of structured data extracted from document parsing is available in `elements.txt` and `split_chunks.txt` files placed in the `structured-data` folder. 
