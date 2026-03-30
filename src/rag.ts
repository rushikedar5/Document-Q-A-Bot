import { pipeline } from "@xenova/transformers";
import { ChromaClient } from "chromadb";
import OpenAI from "openai";
import * as dotenv from "dotenv";

dotenv.config();

async function main() {
const client = new OpenAI({
  apiKey: process.env.GROQ_API_KEY,
  baseURL: "https://api.groq.com/openai/v1",
});

const chroma = new ChromaClient();

const extractor = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
);

async function embed(text: string): Promise<number[]> {
  const output = await extractor(text, { pooling: "mean", normalize: true });
  return Array.from(output.data) as number[];
}

// ── 2. Sample document ───────────────────────────────────────────
const DOCUMENT = `
TypeScript is a strongly typed programming language that builds on JavaScript.
It was developed by Microsoft and first released in 2012.
TypeScript adds optional static typing and class-based object-oriented programming to JavaScript.

The main benefits of TypeScript include catching errors at compile time rather than runtime,
better IDE support with autocomplete and refactoring tools, and improved code readability.

TypeScript code is compiled (or transpiled) to plain JavaScript, which means it runs
anywhere JavaScript runs — browsers, Node.js, and other environments.

Generics in TypeScript allow you to write reusable, type-safe code. For example,
you can write a function that works with any type while still preserving type information.

Interfaces in TypeScript define the shape of an object. They are purely a compile-time
construct and do not exist in the compiled JavaScript output.

TypeScript supports modern JavaScript features like async/await, destructuring,
spread operators, and optional chaining, while also adding its own features like
enums, decorators, and type aliases.
`;

// ── 3. Chunking ───────────────────────────────────────────────────
function chunkText(text: string, chunkSize = 200, overlap = 40): string[] {
  const sentences = text
    .split("\n")
    .map(s => s.trim())
    .filter(s => s.length > 0);

  const chunks: string[] = [];
  let current = "";

  for (const sentence of sentences) {
    if ((current + " " + sentence).length > chunkSize && current.length > 0) {
      chunks.push(current.trim());
      const words = current.split(" ");
      current = words.slice(-overlap / 5).join(" ") + " " + sentence;
    } else {
      current = current ? current + " " + sentence : sentence;
    }
  }

  if (current.trim()) chunks.push(current.trim());
  return chunks;
}

// ── 4. Index document into ChromaDB ─────────────────────────────
async function indexDocument() {
  const collection = await chroma.getOrCreateCollection({
    name: "docs",
    embeddingFunction: null as any,
  });

  // Clear existing data so re-runs don't duplicate chunks
  await collection.delete({ where: { $and: [{}] } }).catch(() => {});
  const existing = await collection.count();
  if (existing > 0) {
    await chroma.deleteCollection({ name: "docs" });
    await chroma.createCollection({
      name: "docs",
      embeddingFunction: null as any,
    });
  }

  const chunks = chunkText(DOCUMENT);
  console.log(`Indexing ${chunks.length} chunks...\n`);

  const embeddings = await Promise.all(chunks.map(embed));

  const freshCollection = await chroma.getOrCreateCollection({
    name: "docs",
    embeddingFunction: null as any,
  });

  await freshCollection.add({
    ids: chunks.map((_, i) => `chunk-${i}`),
    embeddings,
    documents: chunks,
  });

  console.log("Indexed chunks:");
  chunks.forEach((c, i) => console.log(`  [${i}] ${c.substring(0, 60)}...`));
  console.log();
}

// ── 5. Query ──────────────────────────────────────────────────────
async function query(question: string): Promise<string> {
  const collection = await chroma.getOrCreateCollection({
    name: "docs",
    embeddingFunction: null as any,
  });

  const questionVector = await embed(question);
  const results = await collection.query({
    queryEmbeddings: [questionVector],
    nResults: 3,
  });

  const relevantChunks = results.documents[0] ?? [];
  console.log(`Relevant chunks for: "${question}"`);
  relevantChunks.forEach((c, i) => console.log(`  [${i}] ${c?.substring(0, 60)}...`));

  const context = relevantChunks.join("\n\n");

  const response = await client.chat.completions.create({
    model: "llama-3.3-70b-versatile",
    max_tokens: 512,
    messages: [
      {
        role: "system",
        content: `You are a helpful assistant. Answer the user's question using ONLY the context provided below. If the answer is not in the context, say "I don't have that information."

Context:
${context}`,
      },
      { role: "user", content: question },
    ],
  });

  return response.choices[0].message.content ?? "";
}

// ── 6. Run ────────────────────────────────────────────────────────
await indexDocument();

const questions = [
  "Who created TypeScript?",
  "What are the benefits of TypeScript?",
  "What are generics in TypeScript?",
];

for (const q of questions) {
  const answer = await query(q);
  console.log(`Q: ${q}`);
  console.log(`A: ${answer}\n`);
}
}

main().catch(console.error);
