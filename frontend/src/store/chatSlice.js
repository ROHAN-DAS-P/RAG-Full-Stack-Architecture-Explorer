import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  messages: [], // Array of { role: 'user' | 'ai', text: '', citations: [] }
  isGenerating: false,
  error: null,
};

const chatSlice = createSlice({
  name: 'chat',
  initialState,
  reducers: {
    // 1. User sends a message
    addUserMessage: (state, action) => {
      state.messages.push({ role: 'user', text: action.payload });
      // Immediately add an empty AI message placeholder to stream into
      state.messages.push({ role: 'ai', text: '', citations: [] });
      state.isGenerating = true;
      state.error = null;
    },
    // 2. Received citations from backend (happens before text streams)
    addCitations: (state, action) => {
      const lastMessage = state.messages[state.messages.length - 1];
      if (lastMessage && lastMessage.role === 'ai') {
        lastMessage.citations = action.payload;
      }
    },
    // 3. Received a text token from backend
    appendToken: (state, action) => {
      const lastMessage = state.messages[state.messages.length - 1];
      if (lastMessage && lastMessage.role === 'ai') {
        lastMessage.text += action.payload;
      }
    },
    // 4. Stream finished or stopped
    setStreamDone: (state) => {
      state.isGenerating = false;
    },
    // 5. Handle Errors
    setError: (state, action) => {
      state.error = action.payload;
      state.isGenerating = false;
    },
  },
});

export const { addUserMessage, addCitations, appendToken, setStreamDone, setError } = chatSlice.actions;
export default chatSlice.reducer;