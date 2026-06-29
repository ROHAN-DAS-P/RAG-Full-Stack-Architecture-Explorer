import { useRef, useCallback } from "react";
import { useDispatch } from "react-redux";
import {
  addUserMessage,
  addCitations,
  appendToken,
  setStreamDone,
  setError,
} from "../store/chatSlice";

export function useChatWebSocket() {
  const dispatch = useDispatch();
  const wsRef = useRef(null);

  const sendMessage = useCallback(
    (text) => {
      // 1. Update UI immediately
      dispatch(addUserMessage(text));

      // 2. Connect to FastAPI
      const ws = new WebSocket("ws://localhost:8000/ws/chat");
      wsRef.current = ws;

      ws.onopen = () => {
        // 3. Send query matching your backend protocol
        ws.send(JSON.stringify({ type: "query", payload: { text } }));
      };

      // 4. Listen for streaming responses
      ws.onmessage = (event) => {
        const response = JSON.parse(event.data);

        switch (response.type) {
          case "citations":
            dispatch(addCitations(response.payload.citations));
            break;
          case "token":
            dispatch(appendToken(response.payload.text));
            break;
          case "done":
          case "stopped":
            dispatch(setStreamDone());
            ws.close();
            break;
          case "error":
            dispatch(setError(response.payload.message));
            ws.close();
            break;
          default:
            console.warn("Unknown message type:", response.type);
        }
      };

      ws.onerror = () => {
        dispatch(
          setError("WebSocket connection failed. Is the backend running?"),
        );
        dispatch(setStreamDone());
      };
    },
    [dispatch],
  );

  const stopStream = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "stop" }));
    }
  }, []);

  return { sendMessage, stopStream };
}
