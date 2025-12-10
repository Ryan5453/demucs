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
        <div ref={logRef} className="aqua-console">
            {logs.length === 0 ? (
                <div className="aqua-console-entry text-gray-400">Demucs initialized...</div>
            ) : (
                logs.map((log, index) => (
                    <div key={index} className={`aqua-console-entry ${log.type}`}>
                        <span className="aqua-console-timestamp">
                            [{log.timestamp.toLocaleTimeString()}]
                        </span>
                        {log.message}
                    </div>
                ))
            )}
        </div>
    );
}
