import React from "react";
import { createRoot } from "react-dom/client";
import { ProcessArea } from "../../src/components/ProcessArea";
import { MarkdownContent } from "../../src/components/MarkdownContent";
import { answerText, reduceAnswerEvents } from "../../src/run/answerState";
import { reduceRunEvents } from "../../src/run/processState";
import "../../src/App.css";

const root = createRoot(document.getElementById("root"));
window.renderEvents = (events, live) => {
  const answer = reduceAnswerEvents(events);
  root.render(<>
    <ProcessArea process={reduceRunEvents(events)} live={live} />
    <MarkdownContent content={answerText(answer)} />
  </>);
};
