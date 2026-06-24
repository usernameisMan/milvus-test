import "dotenv/config";
import { MilvusClient, MetricType } from "@zilliz/milvus2-sdk-node";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";

const COLLECTION_NAME = "ai_diary";
const VECTOR_DIM = 1024;

// 初始化 OpenAI Chat 模型
const model = new ChatOpenAI({
  temperature: 0.7,
  model: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
});

// 初始化 Embeddings 模型
const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY,
  model: process.env.EMBEDDINGS_MODEL_NAME,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
  dimensions: VECTOR_DIM,
});

const client = new MilvusClient({
  address: "localhost:19530",
});

async function getEmbedding(text) {
  const result = embeddings.embedQuery(text);
  return result;
}

/**
 * 从 Milvus 中检索相关的日记条目
 */
async function retrieveRelevantDiaries(question, k = 2) {
  try {
    // 生成问题的向量
    const queryVector = await getEmbedding(question); // 在 Milvus 中搜索相似的日记

    const searchResult = await client.search({
      collection_name: COLLECTION_NAME,
      vector: queryVector,
      limit: k,
      metric_type: MetricType.COSINE,
      output_fields: ["id", "content", "date", "mood", "tags"],
    });

    return searchResult.results;
  } catch (error) {
    console.error("检索日记时出错:", error.message);
    return [];
  }
}

/**
 *
 * 使用RAG回答日记的问题
 */

async function answerDiaryQuestion(question, k = 2) {
  try {
    console.log("=".repeat(80));
    console.log(`问题: ${question}`);
    console.log("=".repeat(80));

    // 检索相关的日记
    console.log("检索相关的日记...");
    const retrievedDiaries = await retrieveRelevantDiaries(question, k);

    if (retrievedDiaries.length === 0) {
      console.log("没有找到相关的日记");
      return `抱歉,我没有找到相关的日记内容`;
    }

    //2. 答应检索到的日记的相似程度
    retrievedDiaries.forEach((diray) => {
      console.log(`\n[日记 ${diray.id}]`);
      console.log(`相似度：${diray.score.toFixed(4)}`);
      console.log(`日期: ${diray.date}`);
      console.log(`心情: ${diray.mood}`);
      console.log(`标签: ${diray.tags.join(", ")}`);
      console.log(`内容: ${diray.content}`);
    });

    // 3. 构建上下文
    const context = retrievedDiaries
      .map((diary, i) => {
        return `[日记 ${i + 1}]
日期: ${diary.date}
心情: ${diary.mood}
标签: ${diary.tags?.join(", ")}
内容: ${diary.content}`;
      })
      .join("\n\n━━━━━\n\n"); // 4. 构建 prompt

    const prompt = `你是一个温暖贴心的 AI 日记助手。基于用户的日记内容回答问题，用亲切自然的语言。

请根据以下日记内容回答问题：
${context}

用户问题: ${question}

回答要求：
1. 如果日记中有相关信息，请结合日记内容给出详细、温暖的回答
2. 可以总结多篇日记的内容，找出共同点或趋势
3. 如果日记中没有相关信息，请温和地告知用户
4. 用第一人称"你"来称呼日记的作者
5. 回答要有同理心，让用户感到被理解和关心

AI 助手的回答:`; // 5. 调用 LLM 生成回答

    console.log("\n【AI 回答】");
    const response = await model.invoke(prompt);
    console.log(response.content);
    console.log("\n");

    return response.content;
  } catch (error) {
    console.error("回答问题时出错:", error.message);
    return "抱歉，处理您的问题时出现了错误。";
  }
}

async function main() {
  try {
    console.log("Connecting to Milvus...");
    await client.connectPromise;
    console.log("Connected to Milvus");
    await answerDiaryQuestion("我知识上的成长有什么?");
  } catch (error) {
    console.error("回答问题时出错:", error.message);
    return "抱歉，处理您的问题时出现了错误。";
  }
}

main();
