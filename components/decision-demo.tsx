'use client';

import { useState } from 'react';
import { ArrowDown, ArrowRight, Check, ChevronDown, Cpu, RotateCcw } from 'lucide-react';

const examples = [
  { text: 'Set a timer for 10 minutes', values: [3, 80, 17], answer: 'deviceAction' },
  { text: 'Hey, how’s it going?', values: [91, 2, 7], answer: 'casual' },
  { text: 'Explain how photosynthesis works', values: [4, 1, 95], answer: 'assistant' },
];
const options = ['casual', 'deviceAction', 'assistant'];
export function DecisionDemo() {
  const [selected, setSelected] = useState(0);
  const example = examples[selected];
  return <div className="decision-demo">
    <div className="demo-heading"><span><span className="live-dot" /> THE DECISION ENGINE</span><span className="demo-version">KEV / 0.8B</span></div>
    <div className="engine-body">
      <div className="step-label"><span>01</span> STATE <span className="step-hint">Give it context</span></div>
      <div className="state-select"><span className="quote-mark">“</span><select aria-label="Choose an example message" value={selected} onChange={e => setSelected(Number(e.target.value))}>{examples.map((item, i) => <option key={item.text} value={i}>{item.text}</option>)}</select><ChevronDown size={15} /></div>
      <div className="connector"><ArrowDown size={16} /><span>one prefill pass</span></div>
      <div className="step-label"><span>02</span> DECIDE <span className="step-hint">Not generate</span></div>
      <div className="question"><Cpu size={16} /><span>Which route should handle this message?</span></div>
      <div className="probabilities" aria-live="polite">{options.map((option, i) => <div className={`probability ${example.answer === option ? 'winning' : ''}`} key={option}><div className="probability-fill" style={{ width: `${example.values[i]}%` }} /><span className="option-name">{example.answer === option ? <Check size={14} /> : <span className="empty-check" />}{option}</span><span>{(example.values[i] / 100).toFixed(2)}</span></div>)}</div>
      <div className="output"><span className="output-label">OUTPUT</span><code>{example.answer}</code><ArrowRight size={16} /></div>
    </div>
    <div className="demo-footer"><span><span className="live-dot" /> Illustrative example · runs on your device</span><button aria-label="Try next example" onClick={() => setSelected((selected + 1) % examples.length)}><RotateCcw size={14} /></button></div>
  </div>;
}
