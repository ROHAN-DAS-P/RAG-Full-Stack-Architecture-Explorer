import React, { useState, useRef, useEffect } from 'react';
import { useSelector } from 'react-redux';
import { useChatWebSocket } from './hooks/useChatWebSocket';
import { SquareTerminal, Send, StopCircle, FileText, AlertCircle } from 'lucide-react';

export default function App() {
  const [input, setInput] = useState('');
  const { messages, isGenerating, error } = useSelector((state) => state.chat);
  const { sendMessage, stopStream } = useChatWebSocket();
  const [selectedCitation, setSelectedCitation] = useState(null);

  const messagesEndRef = useRef(null);

  // Auto-scroll to the bottom when new tokens stream in
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isGenerating) return;
    sendMessage(input.trim());
    setInput('');
    setSelectedCitation(null); // Clear previous active citation view
  };

  return (
    <div className="flex h-screen w-screen bg-slate-900 text-slate-100 font-sans overflow-hidden">
      
      {/* LEFT SIDE: Chat Interface */}
      <div className="flex flex-col flex-1 h-full border-r border-slate-800">
        
        {/* Header */}
        <header className="flex items-center gap-3 px-6 py-4 border-b border-slate-800 bg-slate-950/50">
          <SquareTerminal className="w-6 h-6 text-emerald-400" />
          <div>
            <h1 className="text-lg font-bold tracking-tight text-slate-200">AI Interview Preparation Assistant</h1>
            <p className="text-xs text-slate-400">
              {/* Local RAG Node running fully offline */}
              </p>
          </div>
        </header>

        {/* Messages Feed */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center max-w-md mx-auto space-y-3">
              <div className="p-4 bg-slate-800/50 rounded-full border border-slate-700/50 text-slate-400">
                <FileText className="w-10 h-10" />
              </div>
              <h3 className="font-semibold text-slate-300">Ask your Documentation</h3>
              <p className="text-sm text-slate-400">
                Type a query below to retrieve context from your local vector database and generate streaming answers.
              </p>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div 
                key={index} 
                className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-2xl px-4 py-3 rounded-2xl border leading-relaxed text-sm ${
                  msg.role === 'user' 
                    ? 'bg-emerald-600 border-emerald-500 text-white rounded-br-none' 
                    : 'bg-slate-950 border-slate-800 text-slate-200 rounded-bl-none'
                }`}>
                  
                  {/* Message Content */}
                  <p className="whitespace-pre-wrap">{msg.text || (isGenerating && index === messages.length - 1 ? '▋' : '')}</p>

                  {/* Citation Pills below AI response */}
                  {msg.role === 'ai' && msg.citations && msg.citations.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-800/60 flex flex-wrap gap-2">
                      <span className="text-xs text-slate-500 block w-full mb-1">Sources retrieved:</span>
                      {msg.citations.map((cite) => (
                        <button
                          key={cite.index}
                          onClick={() => setSelectedCitation(cite)}
                          className={`text-xs px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border text-slate-300 flex items-center gap-1.5 transition-colors cursor-pointer ${
                            selectedCitation?.index === cite.index ? 'border-emerald-500/80 bg-slate-800' : 'border-slate-700/60'
                          }`}
                        >
                          <span className="font-bold text-emerald-400">[{cite.index}]</span>
                          <span className="max-w-[140px] truncate">{cite.source}</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Error Alert Box */}
        {error && (
          <div className="mx-6 my-2 p-3 bg-rose-950/40 border border-rose-900/50 text-rose-200 rounded-xl flex items-start gap-3 text-sm">
            <AlertCircle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
            <p>{error}</p>
          </div>
        )}

        {/* Input Dock */}
        <footer className="p-6 bg-slate-950/30 border-t border-slate-800">
          <form onSubmit={handleSubmit} className="flex gap-2 max-w-4xl mx-auto relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={isGenerating ? "AI is typing..." : "Ask about interview related questions..."}
              disabled={isGenerating}
              className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-emerald-500/80 focus:ring-1 focus:ring-emerald-500/30 disabled:opacity-50 text-slate-200 placeholder-slate-500 transition-all pr-12"
            />
            
            {isGenerating ? (
              <button
                type="button"
                onClick={stopStream}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-lg text-rose-400 hover:bg-slate-800/80 transition-colors cursor-pointer"
                title="Stop generation"
              >
                <StopCircle className="w-5 h-5" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim()}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-lg text-emerald-400 hover:bg-slate-800/80 disabled:text-slate-600 disabled:hover:bg-transparent transition-colors cursor-pointer"
              >
                <Send className="w-5 h-5" />
              </button>
            )}
          </form>
        </footer>
      </div>

      {/* RIGHT SIDE: Dedicated Citation & Metadata Panel */}
      <div className={`w-96 h-full bg-slate-950 flex flex-col transition-all duration-300 ${
        selectedCitation ? 'translate-x-0 opacity-100' : 'translate-x-full w-0 opacity-0'
      }`}>
        {selectedCitation && (
          <div className="flex flex-col h-full p-6 space-y-6 overflow-y-auto">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <h2 className="font-bold text-md text-slate-200 flex items-center gap-2">
                <span className="text-emerald-400 font-extrabold text-lg">[{selectedCitation.index}]</span> 
                Document Snippet
              </h2>
              <button 
                onClick={() => setSelectedCitation(null)}
                className="text-xs px-2 py-1 bg-slate-900 border border-slate-800 rounded text-slate-400 hover:text-slate-200 cursor-pointer"
              >
                Close
              </button>
            </div>

            {/* Source Label Meta */}
            <div className="space-y-3 bg-slate-900/60 border border-slate-800 p-4 rounded-xl text-sm">
              <div className="flex justify-between">
                <span className="text-slate-500">Source File:</span>
                <span className="font-medium text-slate-300 truncate max-w-[160px]">{selectedCitation.source}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Location Reference:</span>
                <span className="font-mono text-emerald-400/90 text-xs">{selectedCitation.label}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Format:</span>
                <span className="text-xs uppercase px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700/50">
                  {selectedCitation.file_type}
                </span>
              </div>
            </div>

            {/* Ingested Chunk Snippet Text */}
            <div className="flex-1 flex flex-col space-y-2">
              <span className="text-xs text-slate-500 font-semibold tracking-wider uppercase">Ground Truth Context</span>
              <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-4 text-xs font-mono leading-relaxed overflow-y-auto text-slate-300 select-all whitespace-pre-wrap">
                {selectedCitation.snippet}
              </div>
            </div>
          </div>
        )}
      </div>

    </div>
  );
}