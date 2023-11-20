package com.rl.poc.langchain4j;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.collections4.CollectionUtils;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.FileSystemDocumentLoader;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.Tokenizer;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiModelName;
import dev.langchain4j.model.openai.OpenAiTokenizer;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

public class SummarizationExample {

	// max amount for demo purposes = 1000 tokens (+- 4000 characters)
	public static final int MAX_TOKEN_LIMIT = 750; // seems like setting a higher token limit is including too many tokens

	// overlap is used when a document is split into multiple sub-document according to the max token limit
	public static final int MAX_OVERLAP_TOKEN_LIMIT = 0;

	public static final ChatLanguageModel CHAT_LANGUAGE_MODEL = OpenAiChatModel.withApiKey("demo");

	public static final PromptTemplate SUMMARIZE_PROMPT_TEMPLATE = PromptTemplate.from("""
			Write a concise summary of the provided text. Maximum one sentence:
			 {{information}}
			SUMMARY:""");

	public static final String QUESTION_WITH_INFORMATION_TEMPLATE = """
			Answer the following question to the best of your ability:
					
			Question:
			{{question}}
					
			Base your answer on the following information:
			{{information}}""";

	private static OpenAiTokenizer OPEN_AI_TOKENIZER = new OpenAiTokenizer(OpenAiModelName.GPT_3_5_TURBO);

	public static void main(String[] args) {
		// Load the document that includes the information you'd like to "chat" about with the model.
		Document document = FileSystemDocumentLoader.loadDocument(toPath("/data/paul_graham_startupideas.txt"));

		List<TextSegment> segments = splitDocument(document, OPEN_AI_TOKENIZER);
		System.out.println(summarizeDocument(segments));

		askQuestionBasedOnDocument(segments);
	}

	private static void askQuestionBasedOnDocument(final List<TextSegment> segments) {
		// Using the document to ask questions:
		// Embed segments (convert them into vectors that represent the meaning) using embedding model
		EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
		List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

		// Store embeddings into embedding store for further search / retrieval
		EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
		embeddingStore.addAll(embeddings, segments);

		// Specify the question you want to ask the model
		String question = "What are the focus points for successful startup ideas?";
		// Embed the question
		Embedding questionEmbedding = embeddingModel.embed(question).content();

		// Find relevant embeddings in embedding store by semantic similarity
		// You can play with parameters below to find a sweet spot for your specific use case
		int maxResults = 1;
		double minScore = 0.7;
		List<EmbeddingMatch<TextSegment>> relevantEmbeddings = embeddingStore.findRelevant(questionEmbedding, maxResults, minScore);

		// Create a prompt for the model that includes question and relevant embeddings
		String information = relevantEmbeddings.stream().map(match -> match.embedded().text()).collect(Collectors.joining("\n\n"));

		Map<String, Object> variables = new HashMap<>();
		variables.put("question", question);
		variables.put("information", information);

		PromptTemplate promptTemplate = PromptTemplate.from(QUESTION_WITH_INFORMATION_TEMPLATE);
		Prompt prompt = promptTemplate.apply(variables);

		// Send the prompt to the chat model
		AiMessage aiMessage = CHAT_LANGUAGE_MODEL.generate(prompt.toUserMessage()).content();

		// See an answer from the model
		System.out.println(aiMessage.text());
	}

	// recursively reduce the document segments to one shorter summary.
	private static String summarizeDocument(List<TextSegment> segments) {
		if (CollectionUtils.isEmpty(segments)) {
			throw new IllegalArgumentException("No text segments provided, can not summarize the document.");
		}

		if (CollectionUtils.size(segments) == 1) { // exit when we have a summary of a single block
			return getSummarization(segments.get(0));
		}

		List<String> summarizations = segments.stream().map(SummarizationExample::getSummarization).toList();

		if (CollectionUtils.size(summarizations) == 1) {
			return summarizations.get(0);
		}

		return summarizeDocument(splitDocument(new Document(String.join("\n", summarizations), new Metadata()), OPEN_AI_TOKENIZER));
	}

	private static String getSummarization(final TextSegment segment) {
		Map<String, Object> variables = new HashMap<>();
		variables.put("information", segment.text());

		// add prompt variables
		Prompt prompt = SUMMARIZE_PROMPT_TEMPLATE.apply(variables);

		// Send the prompt to the chat model
		String summary = CHAT_LANGUAGE_MODEL.generate(prompt.toUserMessage()).content().text();
//		String summary = CHAT_LANGUAGE_MODEL.generate(prompt).content();
		System.out.println(" Partial summary: " + summary);
		return summary;
	}

	private static List<TextSegment> splitDocument(final Document document, final Tokenizer tokenizer) {
		DocumentSplitter splitter = DocumentSplitters.recursive(MAX_TOKEN_LIMIT, MAX_OVERLAP_TOKEN_LIMIT, tokenizer);
		return splitter.split(document);
	}

	private static Path toPath(String fileName) {
		try {
			URL fileUrl = SummarizationExample.class.getResource(fileName);
			return Paths.get(fileUrl.toURI());
		} catch (URISyntaxException e) {
			throw new RuntimeException(e);
		}
	}
}
