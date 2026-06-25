import "dotenv/config"; // 导入并配置 dotenv 以加载环境变量
import { parse, join } from "path"; // 导入 path 模块的 parse 方法，用于解析文件路径
import {
  MilvusClient,
  DataType,
  MetricType,
  IndexType,
} from "@zilliz/milvus2-sdk-node"; // 导入 Milvus 官方 Node.js SDK
import { OpenAIEmbeddings } from "@langchain/openai"; // 导入 LangChain 提供的 OpenAI Embeddings 包装器
import { EPub } from "epub2"; // 导入 epub2 专门解析 EPUB 的 Node 库
import { convert } from "html-to-text"; // 导入 html-to-text 将 HTML 转化为干净纯文本的库
// 【已注释保留】原 LangChain 社区包 EPubLoader 导入方式：
// import { EPubLoader } from "@langchain/community/document_loaders/fs/epub";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"; // 导入递归字符文本拆分器

/**
 * ============================================================================
 * 【教学向】RAG 系统之——电子书流式向量化入库教程
 * ============================================================================
 *
 * 本脚本演示了如何构建一个完整的知识库导入管道（Pipeline），包含以下核心步骤：
 * 1. 【加载】使用 LangChain EPubLoader 读取并解析本地电子书文件（.epub），按章节切割。
 * 2. 【切分】使用递归字符文本切分器（RecursiveCharacterTextSplitter）将长文本拆分为大小合适的片段（Chunk）。
 * 3. 【向量化】通过 OpenAI Embeddings 接口将文本转换为高维实数向量。
 * 4. 【存储】在 Milvus 中创建包含元数据（书名、章节、偏移量）和向量的 Collection，并以章节为单位进行流式批量插入。
 */

// 1. 配置全局参数
const COLLECTION_NAME = "ebook_collection"; // 存储电子书的 Milvus 集合名称
const VECTOR_DIM = 1024; // 向量维度，与使用的 Embedding 模型一致（如 text-embedding-3-large）
const CHUNK_SIZE = 500; // 每一个文本片段（Chunk）的最大字符数
const EPUB_FILE = join(import.meta.dirname, "./天龙八部.epub"); // 要处理的电子书路径

// 从文件名提取书名作为元数据（例如 "./天龙八部.epub" -> "天龙八部"）
const BOOK_NAME = parse(EPUB_FILE).name;

// 2. 初始化 OpenAI Embeddings 工具
// 用于将后续切分出来的中文文本转化为 1024 维度的浮点数向量
const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY, // 接口密钥
  model: process.env.EMBEDDINGS_MODEL_NAME, // 使用的文本嵌入模型名称
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL, // API 基础地址（支持自定义代理或聚合网关）
  },
  dimensions: VECTOR_DIM, // 设置目标向量维度
});

// 3. 初始化 Milvus 客户端实例
const client = new MilvusClient({
  address: "localhost:19530", // Milvus 服务的连接地址与默认 gRPC 端口
});

/**
 * 辅助函数：获取文本的向量嵌入
 *
 * @param {string} text - 需要被向量化的纯文本
 * @returns {Promise<number[]>} - 向量数组（长度为 1024）
 */
async function getEmbedding(text) {
  const result = await embeddings.embedQuery(text);
  return result;
}

/**
 * 确保 Milvus 集合与索引已经建立
 * 类似于在 SQL 数据库中执行 "CREATE TABLE IF NOT EXISTS" 并建立索引
 *
 * @param {string|number} bookId - 电子书的唯一标示 ID
 */
async function ensureCollection(bookId) {
  try {
    // 检查指定的集合是否已存在
    const hasCollection = await client.hasCollection({
      collection_name: COLLECTION_NAME,
    });

    if (!hasCollection.value) {
      console.log("正在创建 Milvus 集合...");

      // 创建集合定义 Schema
      await client.createCollection({
        collection_name: COLLECTION_NAME,
        fields: [
          {
            // id 字段：作为主键（例如：bookId_chapterNum_chunkIndex）
            name: "id",
            data_type: DataType.VarChar,
            max_length: 100,
            is_primary_key: true,
          },
          {
            // 关联的书本 ID
            name: "book_id",
            data_type: DataType.VarChar,
            max_length: 100,
          },
          {
            // 关联的书本名称
            name: "book_name",
            data_type: DataType.VarChar,
            max_length: 200,
          },
          {
            // 当前片段所在的章节号
            name: "chapter_num",
            data_type: DataType.Int32,
          },
          {
            // 片段在当前章节中的序号（从 0 开始）
            name: "index",
            data_type: DataType.Int32,
          },
          {
            // 切分出的文本片段原始内容，由于中文可能较长，最大长度设置为 10000 字符
            name: "content",
            data_type: DataType.VarChar,
            max_length: 10000,
          },
          {
            // 存储向量的字段
            name: "vector",
            data_type: DataType.FloatVector,
            dim: VECTOR_DIM,
          },
        ],
      });
      console.log("✓ 集合创建成功");

      // 向量数据库必须为向量字段创建索引，才能够进行高效的近似邻搜索（ANN）
      console.log("正在创建向量索引...");
      await client.createIndex({
        collection_name: COLLECTION_NAME,
        field_name: "vector",
        index_type: IndexType.IVF_FLAT, // IVF_FLAT 索引：倒排文件索引，适合中等规模数据检索
        metric_type: MetricType.COSINE, // 使用余弦相似度计算向量之间的方向一致性
        params: { nlist: 1024 }, // 聚类中心数量
      });
      console.log("✓ 索引创建成功");
    }

    // 尝试将集合加载到内存中。
    // Milvus 必须加载集合（Load Collection）后才能提供查询与检索服务
    try {
      await client.loadCollection({ collection_name: COLLECTION_NAME });
      console.log("✓ 集合已加载");
    } catch (error) {
      console.log("✓ 集合已处于加载状态");
    }
  } catch (error) {
    console.error("创建集合时出错:", error.message);
    throw error;
  }
}

/**
 * 批量插入文档片段到 Milvus
 * 将当前章节拆分出来的所有文本块，并发转换成向量并打包写入
 *
 * @param {string[]} chunks - 切分好的文本块数组
 * @param {string|number} bookId - 书籍 ID
 * @param {number} chapterNum - 当前章节号
 * @returns {Promise<number>} - 成功插入的记录数
 */
async function insertChunksBatch(chunks, bookId, chapterNum) {
  try {
    if (chunks.length === 0) {
      return 0; // 修复了原代码中 return0 的拼写错误
    }

    // 1. 并发为当前章节的所有文本块生成 Embedding 向量，并组装成 Milvus 要求的行数据结构
    const insertData = await Promise.all(
      // 修复了原代码中 awaitPromise.all 的拼写错误
      chunks.map(async (chunk, chunkIndex) => {
        const vector = await getEmbedding(chunk);
        // 手动生成 ID：bookId_chapterNum_chunkIndex
        return {
          id: `${bookId}_${chapterNum}_${chunkIndex}`,
          book_id: String(bookId),
          book_name: BOOK_NAME,
          chapter_num: chapterNum,
          index: chunkIndex,
          content: chunk,
          vector: vector,
        };
      }),
    );

    // 2. 调用 Milvus client 的 insert 方法，一次性把本章所有数据存入
    const insertResult = await client.insert({
      collection_name: COLLECTION_NAME,
      data: insertData,
    });

    return Number(insertResult.insert_cnt) || 0;
  } catch (error) {
    console.error(`插入章节 ${chapterNum} 的数据时出错:`, error.message);
    console.error("错误详情:", error);
    throw error;
  }
}

/**
 * 核心逻辑：加载 EPUB 并进行流式切分、转换与插入
 * 为什么要采用“边处理边插入（流式处理）”？
 * 一本电子书（如《天龙八部》）有几十万字。如果全部加载到内存中、一次性全部转向量并一次性写入，
 * 会占用极其庞大的内存，且容易因为网络请求并发过高或超时导致服务挂掉。
 * 最佳实践是：按章节遍历，读一章、切一章、转一章、写一章，细水长流。
 *
 * @param {string|number} bookId - 书籍 ID
 */
async function loadAndProcessEPubStreaming(bookId) {
  try {
    console.log(`\n开始加载 EPUB 文件: ${EPUB_FILE}`);

    // 1. 初始化 EPub 实例加载电子书并解析其目录结构
    const epub = await EPub.createAsync(EPUB_FILE);
    console.log(`✓ 加载完成，共 ${epub.flow.length} 个章节\n`);

    /*
    // 【已注释保留】之前使用 LangChain EPubLoader 的加载与拆分章节方式：
    const loader = new EPubLoader(EPUB_FILE, {
      splitChapters: true,
    });
    const documents = await loader.load();
    console.log(`✓ 加载完成，共 ${documents.length} 个章节\n`);
    */

    // 2. 初始化 RecursiveCharacterTextSplitter（递归字符切分器）
    // 它是切分中文的最佳工具。它会根据优先级（如换行符 \n、句号 。、逗号 ，）递归地切分文本，
    // 尽可能保证切分出的 500 字（CHUNK_SIZE）片段保持语句和段落的连贯完整。
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: CHUNK_SIZE, // 每个片段最多包含 500 个字符
      chunkOverlap: 50, // 相邻片段之间重叠 50 个字符，防止章节上下文被拦腰截断时丢失相关语境
    });

    let totalInserted = 0;
    // 3. 循环遍历每一个章节
    for (
      let chapterIndex = 0;
      chapterIndex < epub.flow.length;
      chapterIndex++
    ) {
      const chapter = epub.flow[chapterIndex];

      // 异步读取当前章节的原始 HTML 内容
      const html = await epub.getChapterRawAsync(chapter.id);

      // 使用 html-to-text 过滤 HTML，只保留干净的段落文本，去除图片等非文本标签
      const chapterContent = convert(html, {
        wordwrap: false,
        selectors: [
          { selector: "img", format: "skip" },
          { selector: "a", options: { ignoreHref: true } },
        ],
      });

      /*
      // 【已注释保留】原 LangChain Documents 章节文本获取方式：
      const chapter = documents[chapterIndex];
      const chapterContent = chapter.pageContent;
      */

      console.log(
        `处理第 ${chapterIndex + 1}/${epub.flow.length} 章: ${chapter.title || `第 ${chapterIndex + 1} 章`}...`,
      );

      // 将本章节切分成若干个 500 字的短 Chunk
      const chunks = await textSplitter.splitText(chapterContent);

      console.log(`  拆分为 ${chunks.length} 个片段`);

      if (chunks.length === 0) {
        console.log(`  跳过空章节\n`);
        continue;
      }

      console.log(`  生成向量并插入中...`);

      // 4. 将切好的 Chunk 传入批量写入函数（转向量并存入 Milvus）
      const insertedCount = await insertChunksBatch(
        chunks,
        bookId,
        chapterIndex + 1,
      );
      totalInserted += insertedCount;

      console.log(
        `  ✓ 已插入 ${insertedCount} 条记录（累计: ${totalInserted}）\n`,
      );
    }

    console.log(`\n总共插入 ${totalInserted} 条记录\n`);
    return totalInserted;
  } catch (error) {
    console.error("加载 EPUB 文件时出错:", error.message);
    throw error;
  }
}

/**
 * 流程主入口函数
 */
async function main() {
  try {
    console.log("=".repeat(80));
    console.log("电子书处理程序开始运行...");
    console.log("=".repeat(80));

    // 1. 异步等待 Milvus 连接就绪
    console.log("\n连接 Milvus...");
    await client.connectPromise;
    console.log("✓ 已连接\n");

    // 2. 设定书籍的唯一标识 ID
    const bookId = 1;

    // 3. 准备对应的 Milvus 集合空间与向量索引
    await ensureCollection(bookId);

    // 4. 加载、拆分、向量化并流式插入整本电子书
    await loadAndProcessEPubStreaming(bookId);

    console.log("=".repeat(80));
    console.log("所有数据处理并入库完成！");
    console.log("=".repeat(80));
  } catch (error) {
    console.error("\n错误:", error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

// 启动程序
main();
