# ... (all your existing imports, radar viewer logic, and classes) ...

def main():
    root = TkinterDnD.Tk()
    app = RadarViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
