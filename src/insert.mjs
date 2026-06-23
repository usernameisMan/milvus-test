import "dotenv/config"; // 加载 .env 配置文件中的环境变量
import {
  MilvusClient,
  DataType,
  MetricType,
  IndexType,
} from "@zilliz/milvus2-sdk-node"; // 导入 Milvus SDK 中的相关模块
import { OpenAIEmbeddings } from "@langchain/openai"; // 导入 LangChain OpenAI Embeddings 模块

// 数据库集合（Collection）名称，用于存储 AI 日记
const COLLECTION_NAME = "ai_diary";
// 向量维度大小
const VECTOR_DIM = 1024;

// 初始化 OpenAI Embeddings 实例，用于将文本转换成高维向量
const embeddings = new OpenAIEmbeddings({
  apikey: process.env.OPENAI_API_KEY, // 接口密钥
  model: process.env.EMBEDDINGS_MODEL_NAME, // 嵌入模型名称（如 text-embedding-3-large 等）
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
 * 将传入的文本生成对应的向量（Embedding）
 *
 * @param {string} text - 需要向量化的输入文本
 * @returns {Promise<number[]>} - 包含生成的高维特征向量数组的 Promise
 */
async function getEmbeddings(text) {
  const result = await embeddings.embedQuery(text);
  return result;
}

async function main() {
  try {
    // 1. 连接到 Milvus 数据库
    console.log("Connecting to Milvus...");
    // 调用 connectPromise 进行异步连接，建立与本地 Milvus 服务的 TCP 连接
    await client.connectPromise;

    console.log("✓ Connected\n");

    // 2. 创建集合（Collection），类似于关系型数据库中的表
    console.log("create collection");
    await client.createCollection({
      // 设置集合的名称，在此处为 "ai_diary"
      collection_name: COLLECTION_NAME,
      // 定义集合的结构/Schema（注意原代码中 fields 拼写错误已修正）
      fields: [
        {
          // id 字段：作为每条日记的唯一标识符
          name: "id",
          // 数据类型为 VarChar (字符串)
          data_type: DataType.VarChar,
          // 字符串的最大长度为 50 个字符
          max_length: 50,
          // 设置为主键
          is_primary_key: true,
        },
        {
          // vector 字段：用来存储日记文本生成的特征向量
          name: "vector",
          // 数据类型为浮点向量 FloatVector
          data_type: DataType.FloatVector,
          // 设定向量维度为 1024
          dim: VECTOR_DIM,
        },
        {
          // content 字段：存储日记的纯文本内容
          name: "content",
          // 数据类型为 VarChar (此处原为 Varchar，已修正为 VarChar)
          data_type: DataType.VarChar,
          // 限制日记内容的最大长度为 5000 字符
          max_length: 5000,
        },
        {
          // date 字段：存储日记编写日期
          name: "date",
          // 数据类型为 VarChar (此处原为 Varchar，已修正为 VarChar)
          data_type: DataType.VarChar,
          // 最大长度为 50
          max_length: 50,
        },
        {
          // mood 字段：记录日记的情感/心情标签（如 happy, excited 等）
          name: "mood",
          // 数据类型为 VarChar
          data_type: DataType.VarChar,
          // 最大长度为 50
          max_length: 50,
        },
        {
          // tags 字段：记录日记的自定义标签数组
          name: "tags",
          // 数据类型为 Array (数组类型)
          data_type: DataType.Array,
          // 数组中存储的元素类型为 VarChar (字符串)
          element_type: DataType.VarChar,
          // 每个元素字符串的最大长度为 50
          max_length: 50,
          // 数组的最大容量，最多存储 10 个标签
          max_capacity: 10,
        },
      ],
    });

    console.log("Collection created");

    // 3. 为向量字段创建索引，用以加速后续的向量相似度搜索
    console.log("\nCreaction index...");
    await client.createIndex({
      // 指定在哪个集合上创建索引
      collection_name: COLLECTION_NAME,
      // 需要建索引的字段名，这里是对向量字段 "vector" 建索引
      field_name: "vector",
      // 索引算法类型选择 IVF_FLAT (倒排文件索引)，适合在中等规模的数据集下获得极高的查询速度与不错的精度
      index_type: IndexType.IVF_FLAT,
      // 度量方法选择余弦相似度 MetricType.COSINE，用于计算方向上最相似的文本
      metric_type: MetricType.COSINE,
      params: {
        // IVF_FLAT 的聚类中心参数 nlist，代表将全部向量划分为 1024 个聚类空间
        nlist: 1024,
      },
    });
    console.log("Index created");

    // 4. 加载集合到内存中
    console.log("\nLoading collection...");
    // 在 Milvus 中，所有集合在插入数据后必须先 Load 到内存，才能进行向量检索
    await client.loadCollection({
      collection_name: COLLECTION_NAME,
    });
    console.log("Collection loaded");

    // 5. 准备插入日记数据
    console.log("\nInserting diary entries...");
    // 待插入的原始日记内容列表，包含 id, content, date, mood 以及 tags
    const diaryContents = [
      {
        id: "diary_001",
        content:
          "今天天气很好，去公园散步了，心情愉快。看到了很多花开了，春天真美好。",
        date: "2026-01-10",
        mood: "happy",
        tags: ["生活", "散步"],
      },
      {
        id: "diary_002",
        content:
          "今天工作很忙，完成了一个重要的项目里程碑。团队合作很愉快，感觉很有成就感。",
        date: "2026-01-11",
        mood: "excited",
        tags: ["工作", "成就"],
      },
      {
        id: "diary_003",
        content:
          "周末和朋友去爬山，天气很好，心情也很放松。享受大自然的感觉真好。",
        date: "2026-01-12",
        mood: "relaxed",
        tags: ["户外", "朋友"],
      },
      {
        id: "diary_004",
        content:
          "今天学习了 Milvus 向量数据库，感觉很有意思。向量搜索技术真的很强大。",
        date: "2026-01-12",
        mood: "curious",
        tags: ["学习", "技术"],
      },
      {
        id: "diary_005",
        content:
          "晚上做了一顿丰盛的晚餐，尝试了新菜谱。家人都说很好吃，很有成就感。",
        date: "2026-01-13",
        mood: "proud",
        tags: ["美食", "家庭"],
      },
    ];

    // 6. 并发调用 OpenAI Embeddings 生成向量，并组装成最终待插入的数据对象
    const diaryData = await Promise.all(
      diaryContents.map(async (diary) => ({
        // 复制原有的日记属性 (id, content, date, mood, tags)
        ...diary,
        // 异步获取文本对应的向量特征（注意：原代码中缺失了 await，此处已修正）
        vector: await getEmbeddings(diary.content),
      })),
    );

    // 7. 将包含向量和元数据的日记数据执行批量插入
    const insertResult = await client.insert({
      // 指定插入的集合
      collection_name: COLLECTION_NAME,
      // 待插入的数据列表
      data: diaryData,
    });
    // 打印插入成功的记录数
    console.log(`✓ Inserted ${insertResult.insert_cnt} records \n`);
  } catch (error) {
    // 捕获并输出主函数中的任何报错信息
    console.error("Error in main:", error);
  }
}

main();
