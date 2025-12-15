import { Outlet } from 'react-router-dom';
import Footer from './ui/Footer';

export function Layout() {
    return (
        <>
            <div className="parchment-overlay" />
            <div className="min-h-screen flex flex-col">
                <div className="flex-1 flex flex-col">
                    <Outlet />
                </div>
                <Footer />
            </div>
        </>
    );
}
