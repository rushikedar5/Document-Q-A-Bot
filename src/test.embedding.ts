import { pipeline } from "@xenova/transformers";

async function main() {
  const extractor = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );

  async function embed(text: string): Promise<number[]> {
    const output = await extractor(text, {
      pooling: "mean",
      normalize: true,
    });
    return Array.from(output.data) as number[];
  }

  function cosineSimilarity(a: number[], b: number[]): number {
    const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));

    if (magA === 0 || magB === 0) return 0; // safety check

    return dot / (magA * magB);
  }

  const sentences = [
    "I love dogs",
    "I enjoy puppies",
    "TypeScript is great for backend development",
  ];

  console.log("Generating embeddings (first run downloads the model)...\n");

  const vectors = await Promise.all(sentences.map(embed));

  console.log(`"${sentences[0]}" vs "${sentences[1]}":`);
  console.log(
    "Similarity:",
    cosineSimilarity(vectors[0], vectors[1]).toFixed(4)
  );

  console.log(`\n"${sentences[0]}" vs "${sentences[2]}":`);
  console.log(
    "Similarity:",
    cosineSimilarity(vectors[0], vectors[2]).toFixed(4)
  );
}

// 🚀 run everything
main().catch(console.error);