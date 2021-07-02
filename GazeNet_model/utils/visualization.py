import win32gui
import win32con
import win32api
import colorsys
import time
import tkinter as tk

def move_win(pos):
    window = tk.Tk()
    # On the top
    window.attributes("-topmost", True)
    window.attributes("-alpha", 0.7)
    window.overrideredirect(True)
    # Fill with red
    Full_color = tk.Label(window, bg='red', width=480, height=270)
    Full_color.pack()
    window.geometry("%dx%d+%d+%d" % (480, 270, pos[0] - 240, pos[1] - 135))
    # Update the window
    window.update()
    # time.sleep(1)
    window.destroy()

def posdef(classes):
    if classes == 0:
        return (480, 270)
    if classes == 1:
        return (480, 540)
    if classes == 2:
        return (480, 810)
    if classes == 3:
        return (960, 270)
    if classes == 4:
        return (960, 540)
    if classes == 5:
        return (960, 810)
    if classes == 6:
        return (1440, 270)
    if classes == 7:
        return (1440, 540)
    if classes == 8:
        return (1440, 810)

def visual(pred):
    cord = posdef(pred)
    return cord

# image frame
def drawRect(pos, pred, colors):
    hwnd = win32gui.GetDesktopWindow()
    # define the colors
    hPen = win32gui.CreatePen(win32con.PS_SOLID, 3, win32api.RGB(colors[pred][0], colors[pred][1], colors[pred][2]))
    win32gui.InvalidateRect(hwnd, None, True)
    win32gui.UpdateWindow(hwnd)
    win32gui.RedrawWindow(hwnd, None, None,
                          win32con.RDW_FRAME | win32con.RDW_INVALIDATE | win32con.RDW_UPDATENOW | win32con.RDW_ALLCHILDREN)
    # obtain the device context DC (Divice Context) of the window according to the window handle
    hwndDC = win32gui.GetDC(hwnd)

    win32gui.SelectObject(hwndDC, hPen)
    # define the transparent brush, this is very important!
    hbrush = win32gui.GetStockObject(win32con.NULL_BRUSH)
    prebrush = win32gui.SelectObject(hwndDC, hbrush)
    # coordinates from top left to bottom right
    win32gui.Rectangle(hwndDC, pos[0] - 240, pos[1] - 135, pos[0] + 240, pos[1] + 135)
    win32gui.SaveDC(hwndDC)
    win32gui.SelectObject(hwndDC, prebrush)
    # recycling the resources
    win32gui.DeleteObject(hPen)
    win32gui.DeleteObject(hbrush)
    win32gui.DeleteObject(prebrush)
    win32gui.ReleaseDC(hwnd, hwndDC)


if __name__ == '__main__':
    hsv_tuples = [(x / 9, 1., 1.)
                  for x in range(9)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    # pred = 1
    for pred in range(0, 9):
        cord = visual(pred)
        #drawRect(cord, pred, colors)
        #time.sleep(2)
        move_win(cord)


