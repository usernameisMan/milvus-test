import "dotenv/config"; // 加载 .env 配置文件中的环境变量
import { MilvusClient, DataType, MetricType } from "@zilliz/milvus2-sdk-node"; // 导入 Milvus 客户端及相关类型定义
import { OpenAIEmbeddings } from "@langchain/openai"; // 导入 LangChain OpenAI Embeddings 模块

// 数据库集合名称
const COLLECTION_NAME = "ai_diary";
// 向量维度大小
const VECTOR_DIM = 1024;

// 初始化 OpenAI Embeddings 实例，用于将更新后的文本转换成特征向量
const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY, // 从环境变量中读取 OpenAI API Key
  model: process.env.EMBEDDINGS_MODEL_NAME, // 从环境变量中读取嵌入模型的名称
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL, // 支持自定义的 OpenAI API 基础 URL
  },
  dimensions: VECTOR_DIM, // 设置目标向量维度
});

// 初始化 Milvus 客户端并连接到本地的 Milvus 服务
const client = new MilvusClient({
  address: "localhost:19530",
});

/**
 * 将传入的文本生成对应的向量（Embedding）
 *
 * @param {string} text - 需要向量化的输入文本
 * @returns {Promise<number[]>} - 返回文本对应的浮点数向量数组
 */
async function getEmbedding(text) {
  const result = await embeddings.embedQuery(text);
  return result;
}

async function main() {
  try {
    console.log(`Connection to Milvus..`);
    // client.connectPromise 是一个 Promise 属性（Getter），代表后台正在进行的连接过程。
    // 因为它直接是一个 Promise 实例，而不是一个返回 Promise 的方法，所以不需要加括号 ()，直接 await 即可。
    await client.connectPromise;
    console.log(`Connected`);

    // 更新数据 (Milvus 通过 upsert 方法实现更新/插入操作：若 ID 存在则更新，不存在则插入)
    console.log("Updating diary entry ...");
    const updateId = `diary_001`; // 准备更新的日记唯一 ID
    const updatedContent = {
      id: updateId,
      content:
        "今天下了一整天的雨，心情很糟糕。工作上遇到了很多困难，感觉压力很大。一个人在家，感觉特别孤独。",
      date: "2026-01-10",
      mood: "sad",
      tags: ["生活", "散步", "朋友"],
    };

    console.log("Generating new emdedding...");
    // 异步生成更新后内容的向量
    const vector = await getEmbedding(updatedContent.content);

    // 将原字段与新生成的向量重新组合成完整的待插入数据对象
    const updateData = { ...updatedContent, vector };

    // 执行 upsert 操作
    const result = await client.upsert({
      collection_name: COLLECTION_NAME,
      data: [updateData],
    });

    // 打印更新成功后的相关元数据
    console.log(`✓ Updated diary entry: ${updateId}`);
    console.log(`  New content: ${updatedContent.content}`);
    console.log(`  New mood: ${updatedContent.mood}`);
    console.log(`  New tags: ${updatedContent.tags.join(", ")}\n`);
  } catch (error) {
    // 捕获并输出更新过程中的异常
    console.error("Error:", error.message);
  }
}

// 执行更新主函数
main();
