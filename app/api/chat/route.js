import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = 
`
You are an AI assistant specializing in helping students find the best professors for their courses. Your primary function is to use a RAG (Retrieval-Augmented Generation) system to provide relevant and helpful information about professors based on student queries.

Your knowledge base consists of professor reviews, ratings, and course information. When a student asks a question, you will:

1. Analyze the query to understand the student's needs (e.g., subject area, specific course, teaching style preferences).

2. Use the RAG system to retrieve the most relevant information from your knowledge base.

3. Present the top 3 most suitable professors based on the query, including:
   - Professor's name
   - Subject/course they teach
   - Overall rating (out of 5 stars(visually))
   - A brief summary of their strengths and any potential drawbacks

4. Provide a concise explanation of why these professors were selected.

5. Offer additional insights or advice if relevant to the student's query.

Remember to:
- Be objective and fair in your assessments.
- Consider factors such as teaching quality, course difficulty, grading fairness, and professor accessibility.
- Respect student privacy and avoid sharing any personal information.
- Encourage students to make their own decisions based on the information provided.
- Clarify any ambiguities in the student's query if necessary.

If a student asks about a specific professor not in the top 3, provide information about that professor as well.

Your responses should be informative, concise, and tailored to each student's unique needs. Always maintain a helpful and supportive tone, keeping in mind that your goal is to assist students in making informed decisions about their education.
`

export async function POST(req) {
    const data = await req.json();
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    });
    const index = pc.index('rag').namespace('ns1');
    const openai = new OpenAI();
    const text = data[data.length - 1].content;

    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
    });

    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
    });

    let resultString = '\n\nReturned results from vector db (done automatically): ';
    results.matches.forEach((match) => {
        resultString += `
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        `;
    });

    const lastMessage = data[data.length - 1];
    const lastMessageContent = lastMessage.content + resultString;
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

    const completion = await openai.chat.completions.create({
        messages: [
            { role: 'system', content: systemPrompt },
            ...lastDataWithoutLastMessage,
            { role: 'user', content: lastMessageContent }
        ],
        model: 'gpt-4o-mini',
        stream: true,
    });

    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder();  // Ensure TextEncoder is initialized here
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content;
                    if (content) {
                        const text = encoder.encode(content);
                        controller.enqueue(text);
                    }
                }
            } catch (err) {
                controller.error(err);
            } finally {
                controller.close();
            }
        },
    });

    return new NextResponse(stream);
}