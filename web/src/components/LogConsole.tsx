import { useRef, useEffect } from 'react';
import type { LogEntry } from '../types';

interface LogConsoleProps {
    logs: LogEntry[];
}

export function LogConsole({ logs }: LogConsoleProps) {
    const logRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (logRef.current) {
            logRef.current.scrollTop = logRef.current.scrollHeight;
        }
    }, [logs]);

    return (
        <div ref={logRef} className="log-console">
            {logs.length === 0 ? (
                <div className="log-entry text-gray-600">Demucs initialized...</div>
            ) : (
                logs.map((log, index) => (
                    <div key={index} className={`log-entry ${log.type}`}>
                        [{log.timestamp.toLocaleTimeString()}] {log.message}
                    </div>
                ))
            )}
        </div>
    );
}
