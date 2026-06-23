import "dotenv/config"; // 加载 .env 配置文件中的环境变量
import { MilvusClient, MetricType } from "@zilliz/milvus2-sdk-node"; // 导入 Milvus 客户端及度量类型定义
import { OpenAIEmbeddings } from "@langchain/openai"; // 导入 LangChain OpenAI Embeddings 模块

// 数据库集合名称
const COLLECTION_NAME = "ai_diary";
// 向量的维度大小，使用 1024 维
const VECTOR_DIM = 1024;

// 初始化 OpenAI Embeddings 实例，用于将查询文本转换成向量
const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY, // 接口密钥
  model: process.env.EMBEDDINGS_MODEL_NAME, // 嵌入模型名称
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL, // OpenAI API 基础地址（支持自定义代理）
  },
  dimensions: VECTOR_DIM, // 设定的向量维度
});

// 初始化 Milvus 客户端，默认连接本地 19530 端口
const client = new MilvusClient({
  address: "localhost:19530",
});

/**
 * 将输入的文本生成对应的特征向量
 *
 * @param {string} text - 输入查询文本
 * @returns {Promise<number[]>} - 包含生成的高维特征向量数组的 Promise
 */
async function getEmbedding(text) {
  const result = await embeddings.embedQuery(text);
  return result;
}

async function main() {
  try {
    // 1. 连接到 Milvus 数据库
    console.log("Connecting to Milvus...");
    // 异步等待与 Milvus 建立连接的 Promise 完成
    await client.connectPromise;
    console.log("✓ Connected\n");

    // 2. 定义搜索关键字并打印
    console.log("Searching for similar diary entries...");
    const query = "我想看看关于提升自己知识的日记";
    console.log(`Query: "${query}"\n`);

    // 3. 将搜索关键字转化为 1024 维的查询向量
    const queryVector = await getEmbedding(query);

    // 4. 调用 search 方法执行向量相似度检索
    const searchResult = await client.search({
      // 目标查询集合
      collection_name: COLLECTION_NAME,
      // 输入的查询向量（刚刚生成的 1024 维特征向量）
      vector: queryVector,
      // 限制返回相似度最高的前 2 条记录
      limit: 2,
      // 相似度度量标准，使用余弦相似度 (COSINE)。余弦相似度分数越大，表示方向越一致，内容越相似。
      metric_type: MetricType.COSINE,
      // 查询结果中需要返回的其他属性字段（包括元数据如日记内容、时间、情绪、标签等）
      output_fields: ["id", "content", "date", "mood", "tags"],
    });

    // 5. 格式化输出查询到的相似日记结果
    console.log(`Found ${searchResult.results.length} results:\n`);
    searchResult.results.forEach((item, index) => {
      // 打印匹配序号与相似度得分 (Score)
      console.log(`${index + 1}. [Score: ${item.score.toFixed(4)}]`);
      console.log(`   ID: ${item.id}`);
      console.log(`   Date: ${item.date}`);
      console.log(`   Mood: ${item.mood}`);
      // 格式化输出 tags 数组
      console.log(`   Tags: ${item.tags?.join(", ")}`);
      console.log(`   Content: ${item.content}\n`);
    });
  } catch (error) {
    // 捕获异常并打印错误信息
    console.error("Error:", error.message);
  }
}

// 执行查询主函数
main();
