import type { ReactNode } from 'react';

interface Column<T> {
    key: string;
    header: string;
    width?: string;
    render?: (item: T, index: number) => ReactNode;
}

interface AquaTableProps<T> {
    columns: Column<T>[];
    data: T[];
    keyExtractor: (item: T) => string;
    selectedKey?: string;
    onRowClick?: (item: T) => void;
    emptyMessage?: string;
}

export function AquaTable<T>({
    columns,
    data,
    keyExtractor,
    selectedKey,
    onRowClick,
    emptyMessage = 'No items'
}: AquaTableProps<T>) {
    if (data.length === 0) {
        return (
            <div className="aqua-inset-panel text-center py-8">
                <span className="text-sm text-gray-500">{emptyMessage}</span>
            </div>
        );
    }

    return (
        <table className="aqua-table">
            <thead>
                <tr>
                    {columns.map((col) => (
                        <th key={col.key} style={{ width: col.width }}>
                            {col.header}
                        </th>
                    ))}
                </tr>
            </thead>
            <tbody>
                {data.map((item, index) => {
                    const key = keyExtractor(item);
                    const isSelected = key === selectedKey;
                    return (
                        <tr
                            key={key}
                            className={isSelected ? 'selected' : ''}
                            onClick={() => onRowClick?.(item)}
                            style={{ cursor: onRowClick ? 'pointer' : 'default' }}
                        >
                            {columns.map((col) => (
                                <td key={col.key}>
                                    {col.render
                                        ? col.render(item, index)
                                        : String((item as Record<string, unknown>)[col.key] ?? '')
                                    }
                                </td>
                            ))}
                        </tr>
                    );
                })}
            </tbody>
        </table>
    );
}
