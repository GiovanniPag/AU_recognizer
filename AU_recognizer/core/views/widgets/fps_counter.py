import tkinter as tk


class FPSCounter:
    def __init__(self, parent):
        """Create a transparent FPS counter overlay."""
        self.parent = parent
        self.hidden = False

        # Create a transparent frame for the FPS counter
        self.frame = tk.Frame(parent, bg="black", highlightthickness=0)
        self.frame.place(x=0, y=0)

        # FPS label
        self.fps_label = tk.Label(
            self.frame,
            text="FPS: 0",
            font=("Arial", 10),
            bg="black",
            fg="white",
            anchor="w"
        )
        self.fps_label.pack(side="left", padx=5)

        # Close button (appears on hover)
        self.close_button = tk.Button(
            self.frame,
            text="x",
            font=("Arial", 10),
            bg="black",
            fg="red",
            borderwidth=0,
            command=self.hide
        )

        # Show close button on hover
        self.frame.bind("<Enter>", self.show_close_button)
        self.frame.bind("<Leave>", self.hide_close_button)

    def update(self, fps):
        """Update the FPS label."""
        self.fps_label.config(text=f"FPS: {fps:.2f}")

    def hide(self):
        """Hide the FPS counter."""
        self.hidden = True
        self.frame.place_forget()

    def show(self):
        """Show the FPS counter."""
        self.hidden = False
        self.frame.place(x=0, y=0)

    def toggle(self):
        """toggle the FPS counter."""
        if self.hidden:
            self.show()
        else:
            self.hide()

    def show_close_button(self, event=None):
        """Show the close button."""
        self.close_button.pack(side="right", padx=5)

    def hide_close_button(self, event=None):
        """Hide the close button."""
        self.close_button.pack_forget()
