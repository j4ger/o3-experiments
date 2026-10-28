package dev.j4ger.o3chat;

import android.graphics.Bitmap;
import android.os.Binder;
import android.os.IBinder;
import android.os.Parcel;
import android.os.RemoteException;

/**
 * Client-side implementation of com.xiaomi.taiyi.sdk.transfer.data.IBitmap.
 *
 * AIPackage.putBitmap(..., useBinder=true) stores an IBitmap Binder, not a raw
 * Bitmap.  The AIService plugin calls IBitmap.get() while building the VLM
 * prompt, so the bitmap has to be returned from a Binder transaction.
 */
public final class BitmapBinder extends Binder {
    public static final String DESCRIPTOR = "com.xiaomi.taiyi.sdk.transfer.data.IBitmap";

    private final Bitmap bitmap;

    public BitmapBinder(Bitmap bitmap) {
        this.bitmap = bitmap;
    }

    @Override
    protected boolean onTransact(int code, Parcel data, Parcel reply, int flags)
            throws RemoteException {
        if (code == IBinder.INTERFACE_TRANSACTION) {
            if (reply != null) {
                reply.writeString(DESCRIPTOR);
            }
            return true;
        }
        if (code != 1) {
            return super.onTransact(code, data, reply, flags);
        }

        data.enforceInterface(DESCRIPTOR);
        if (reply == null) {
            return true;
        }
        reply.writeNoException();
        if (bitmap == null || bitmap.isRecycled()) {
            reply.writeInt(0);
        } else {
            reply.writeInt(1);
            bitmap.writeToParcel(reply, 0);
        }
        return true;
    }
}
