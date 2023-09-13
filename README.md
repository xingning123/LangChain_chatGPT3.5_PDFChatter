# LangChain_chatGPT3.5_PDFChatter

This is a project for extracting sth in a pdf file. Use langchain; gpt3.5; PaoDing AI; prompt engineering



项目描述：
本项目旨在从PDF文档中提取关键信息，并将其结构化为直观的CSV格式。为了实现这一目标，我们采用了以下方法：

1. 使用OpenAI GPT-3.5 API：利用GPT-3.5的强大能力，我们创建了一个问答对象，该对象可以根据自定义提示从文档中提取特定信息。
  
2. 文档加载与处理：使用Langchain的PyPDFLoader工具包加载PDF文档，并使用CharacterTextSplitter工具包将文档切分为更小的部分，以便于后续处理。
  
3. 文档向量化：使用OpenAIEmbeddings工具包将切分后的文档转化为嵌入向量，并存储在Chroma向量数据库中，以便后续的匹配查询。
  
4. 信息提取：根据自定义的提示模板，我们查询了文档中的特定信息，如建筑地址、批次和材料。为了确保提取的信息是准确的，我们设计了一个循环，直到找到满足条件的答案为止。
  
5. 结构化输出：最后，我们将提取的信息保存为JSON和CSV格式，以便于后续的分析和使用。
