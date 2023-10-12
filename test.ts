// Test multipass prompts taken from https://github.com/raidendotai/openv0/tree/main/server/modules/multipass/passes

import { OpenAI } from "openai";

import { model } from "./component-designer/model.js";

async function run() {
  const openai = new OpenAI();

  const multipass = model.build(openai);

  const generated = await multipass({
    input:
      "A simple hero section with a title, subtitle, and CTA button that's actually an anchor.",
  });

  console.log("Generated:");
  console.log("NAME:", generated.name);
  console.log("DESCRIPTION:", generated.description);
  console.log("CODE:");
  console.log(generated.code);
}

run().catch(console.error);
