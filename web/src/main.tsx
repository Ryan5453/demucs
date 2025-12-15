import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { DemucsApp } from './components/DemucsApp'
import { About } from './pages/About'
import { Privacy } from './pages/Privacy'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<DemucsApp />} />
                <Route path="/about" element={<About />} />
                <Route path="/privacy" element={<Privacy />} />
            </Routes>
        </BrowserRouter>
    </React.StrictMode>,
)
