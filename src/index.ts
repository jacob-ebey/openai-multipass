import type { OpenAI } from "openai";

interface CompleteArgs {
  model: "gpt-4" | "gpt-3.5-turbo";
  functions?: OpenAI.ChatCompletionCreateParams["functions"];
  messages: {
    role: "function" | "system" | "user" | "assistant";
    content: string;
  }[];
}

interface Completion {
  functionCall?: unknown;
  content?: string;
}

export interface PassFuncArgs<I> {
  input: I;
  complete(args: CompleteArgs): Promise<Completion>;
  embed(text: string): Promise<Array<number[]>>;
}

export type PassFunction<I, R> = (args: PassFuncArgs<I>) => R;

export interface MultipassFactory<I = string, R = unknown, PI = I> {
  pass<R>(
    name: string,
    func: PassFunction<PI, R>
  ): MultipassFactory<I, Awaited<R>, Awaited<R>>;
  build(openai: OpenAI): Multipass<I, R>;
}

export interface MultipassArgs<I> {
  input: I;
}

export type Multipass<I, T> = (args: MultipassArgs<I>) => Promise<T>;

export function multipassFactory({
  debug,
}: { debug?: boolean } = {}): MultipassFactory {
  const passes: { name: string; func: PassFunction<unknown, unknown> }[] = [];
  const factory = {
    pass<R>(name, func) {
      passes.push({ name, func });

      return factory;
    },
    build(openai: OpenAI) {
      const complete = createComplete(openai);
      const embed = createEmbed(openai);
      return (async ({ input }: MultipassArgs<unknown>) => {
        for (const pass of passes) {
          if (debug) {
            console.debug(`[MULTIPASS] running pass ${pass.name}`);
          }
          input = await pass.func({ input, complete, embed });
        }

        return input;
      }) as any;
    },
  } satisfies MultipassFactory;

  return factory;
}

function createComplete(
  openai: OpenAI
): (args: CompleteArgs) => Promise<Completion> {
  return async (args) => {
    const stream = await openai.chat.completions.create({
      model: args.model,
      messages: args.messages,
      functions: args.functions,
      stream: true,
    });

    let content = "";
    let functionCall = "";
    for await (const part of stream) {
      content += part.choices[0]?.delta?.content || "";
      functionCall += part.choices[0]?.delta?.function_call?.arguments || "";
    }

    return {
      functionCall: functionCall ? JSON.parse(functionCall) : undefined,
      content: content ? content : undefined,
    };
  };
}

function createEmbed(
  openai: OpenAI
): (text: string) => Promise<Array<number[]>> {
  return async (text) => {
    const response = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: text,
    });

    return response.data.map((d) => d.embedding);
  };
}
