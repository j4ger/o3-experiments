package com.xiaomi.taiyi.sdk.transfer.data;

import android.os.Bundle;
import android.os.Parcel;
import android.os.Parcelable;

/**
 * Client-side stub with the exact class name and parcel layout of AIService's
 * AIHeaders.  The server reads this parcelable with its own class loader, so
 * only the class name and bundle layout have to match.
 */
public final class AIHeaders implements Parcelable {
    public Bundle bundle = new Bundle();

    public AIHeaders() {
    }

    private AIHeaders(Parcel in) {
        readFromParcel(in);
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeBundle(bundle);
    }

    public void readFromParcel(Parcel in) {
        Bundle b = in.readBundle(AIHeaders.class.getClassLoader());
        bundle = b != null ? b : new Bundle();
        bundle.setClassLoader(AIHeaders.class.getClassLoader());
    }

    @Override
    public int describeContents() {
        return 0;
    }

    public static final Parcelable.Creator<AIHeaders> CREATOR =
            new Parcelable.Creator<AIHeaders>() {
                @Override
                public AIHeaders createFromParcel(Parcel in) {
                    return new AIHeaders(in);
                }

                @Override
                public AIHeaders[] newArray(int size) {
                    return new AIHeaders[size];
                }
            };
}
