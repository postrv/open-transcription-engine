// File: transcription_engine/static/src/components/Header.jsx
import React from 'react';
import { Button } from './ui/button';
import { MoonIcon, SunIcon, MenuIcon } from 'lucide-react';

const Header = ({ theme, toggleTheme }) => {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="hidden md:block">
            <h1 className="text-xl font-semibold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
              Transcription Engine
            </h1>
          </div>
          <div className="block md:hidden">
            <Button variant="ghost" size="icon">
              <MenuIcon className="h-5 w-5" />
            </Button>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <nav className="hidden md:flex items-center space-x-4">
            <Button variant="ghost" className="text-sm font-medium">
              Dashboard
            </Button>
            <Button variant="ghost" className="text-sm font-medium">
              Transcripts
            </Button>
            <Button variant="ghost" className="text-sm font-medium">
              Settings
            </Button>
          </nav>

          <div className="flex items-center space-x-2 border-l pl-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleTheme}
              className="h-9 w-9 transition-transform hover:rotate-45"
            >
              {theme === 'light' ? (
                <SunIcon className="h-4 w-4" />
              ) : (
                <MoonIcon className="h-4 w-4" />
              )}
              <span className="sr-only">Toggle theme</span>
            </Button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
